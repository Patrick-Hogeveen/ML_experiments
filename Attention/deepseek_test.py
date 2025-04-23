import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import math
from pickle import dump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.vocab_size = 32000 #Size of tokenizers vocabulary Adjust this to 65 for the shakespeare test
        self.d_model = 5120 #Hidden dimension of the model
        self.n_layers = 2 #Number of transoformer blocks
        self.n_heads = 8 #Number of attention heads
        self.d_kv_comp = 128 #Latent dimesnion for compressed keys/values
        self.d_rope = 16 #rotary embedding dimension applied to a subset of query/key heads
        self.n_experts = 32 #Total number of routed experts
        self.n_shared = 2 #number of always active experts
        self.top_k = 2 #Number of experts activated per token
        self.seq_len = 256 #Maximum length of sequence during training
        self.batch_size = 1 #number of sequences to process in parallel
        self.ffn_dim = 384 #hidden dimension of feed foward network
        self.device_groups = 4 # For device-limited routing

config = Config()

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale=40):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = 40

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x, cos, sin):
    """
    Apply rotary embeddings to the first half of x.
    """
    # Split x into two parts: one for rotary embeddings and the other untouched
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    # Apply rotary embeddings to the rotary part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # Concatenate the rotary-applied and base parts
    return torch.cat([x_rot, x_base], dim=-1)

class MemoryOptimizedMLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_head = config.d_model // config.n_heads
        self.split_dim = self.d_head - config.d_rope   

        # Projections
        self.W_dkv = nn.Linear(config.d_model, config.d_kv_comp)
        self.W_dq = nn.Linear(config.d_model, config.d_kv_comp)

        self.W_uk = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)
        self.W_uv = nn.Linear(config.d_kv_comp, config.n_heads * self.d_head)  
        self.W_uq = nn.Linear(config.d_kv_comp, config.n_heads * self.split_dim)

        self.W_qr = nn.Linear(config.d_kv_comp, config.n_heads * config.d_rope)
        self.W_kr = nn.Linear(config.d_model, config.n_heads * config.d_rope)

        self.rotary = RotaryEmbedding(config.d_rope)
        self.output = nn.Linear(config.n_heads * self.d_head, config.d_model)

    def forward(self, h, past_kv=None):
        batch_size, seq_len, _ = h.shape

        # KV Compression
        c_kv = self.W_dkv(h)
        k = self.W_uk(c_kv).view(batch_size, seq_len, config.n_heads, self.split_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, config.n_heads, self.d_head)

        # Query Compression
        c_q = self.W_dq(h)
        q_base = self.W_uq(c_q).view(batch_size, seq_len, config.n_heads, self.split_dim)
        q_rot = self.W_qr(c_q).view(batch_size, seq_len, config.n_heads, config.d_rope)

        # Rotary embeddings with proper dimensions
        rotary_emb = self.rotary(seq_len)
        cos = torch.cos(rotary_emb).view(1, seq_len, 1, -1)  # [1, seq, 1, dim]
        sin = torch.sin(rotary_emb).view(1, seq_len, 1, -1)

        # Apply rotary embeddings
        q_rot = apply_rotary(q_rot, cos, sin)
        k_rot = apply_rotary(
            self.W_kr(h).view(batch_size, seq_len, config.n_heads, config.d_rope),
            cos, sin
        )

        q = torch.cat([q_base, q_rot], dim=-1)
        k = torch.cat([k, k_rot], dim=-1)

        # Attention computation
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)

        return self.output(out.contiguous().view(batch_size, seq_len, -1)), (c_kv, k_rot)
    
class DeepSeekMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts = nn.ModuleList([Expert() for _ in range(config.n_shared)])
        self.routed_experts = nn.ModuleList([Expert() for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.d_model, config.n_experts)
        self.aux_loss = 0.0
    
    def forward(self, x):
        # Shared experts process all tokens
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Device-limited routing
        routed_logits = self.gate(x)
        probs = F.softmax(routed_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(config.top_k, dim=-1)

        # Expert balance loss
        expert_counts = torch.zeros(config.n_experts, device=x.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1),
                                 torch.ones_like(topk_indices.view(-1), dtype=torch.float))
        self.aux_loss += expert_counts.float().var() * 0.003  # α1 from paper

        # Sparse computation
        routed_out = torch.zeros_like(x)
        for k in range(config.top_k):
            expert_mask = topk_indices[..., k]
            expert_contrib = torch.zeros_like(x)

            for expert_idx in range(config.n_experts):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * topk_probs[..., k][mask].unsqueeze(-1)

            routed_out += expert_contrib

        return shared_out + routed_out
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MemoryOptimizedMLA()
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = DeepSeekMoE()

    def forward(self, x, past_kv=None):
        # Attention with KV cache
        attn_out, new_kv = checkpoint(self.attn, self.norm1(x), past_kv)
        x = x + attn_out

        # MoE with checkpointing
        moe_out = checkpoint(self.moe, self.norm2(x))
        x = x + moe_out

        return x, new_kv
    
class DeepSeekV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Better initialization with residual scaling
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1/math.sqrt(config.n_layers))
        # Add residual scaling
        for block in self.blocks:
            block.attn.output.weight.data.mul_(0.1)
            block.moe.shared_experts[0].w2.weight.data.mul_(0.1)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux_loss = 0.0

        for block in self.blocks:
            x, _ = block(x)
            total_aux_loss += block.moe.aux_loss

        return self.lm_head(self.norm(x)), total_aux_loss
    
    def generate(self, x, max_length):

        for _ in range(max_length):
            x_cond = x[:, -config.seq_len:]
            logits, loss = self(x_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x,x_next), dim=1)

        return x

    
def train(model):
    model = model.half()
    model = model.to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    optimizer_dict = {p: torch.optim.Adam([p], foreach=False, eps=10e-4) for p in model.parameters()}

    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)
    
    #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #    optimizer_dict,
    #    max_lr=3e-4,
    #    total_steps=40,
    #    pct_start=0.1,
    #)

    for epoch in range(40):
        
        xb, yb = get_batch('train')

        #optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            logits, aux_loss = model(xb[:,:-1])
            loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                   xb[:,1:].contiguous().view(-1))
            loss += 0.0001 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        #optimizer.step()
        #lr_scheduler.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
    
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

torch.manual_seed(1337)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.seq_len, (config.batch_size,))
    x = torch.stack([data[i:i+config.seq_len+1] for i in ix])
    y = torch.stack([data[i+1:i+config.seq_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

if __name__ == '__main__' :
    model = DeepSeekV2().to(device)
    model.load_state_dict(torch.load('mini_deepseek'))

    inp = "wherefore art thou"
    inp = torch.tensor(encode(inp)).unsqueeze(0).to(device)
    print(inp[0].shape)
    output = model.generate(inp, max_length=500)
    print(output.shape)
    print(decode(output[0].tolist()))