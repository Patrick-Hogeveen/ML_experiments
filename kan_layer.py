import torch
import numpy as np
from spline import *
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(1)

class kan_layer(nn.Module):
    def __init__(self, indim=3, outdim=3, grid_size=5, spline_order=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, stand_alone_spline_scale=False, base_func=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1,1],sp_trainable=True, sb_trainable=True, device='cuda'):
        super(kan_layer, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.grid_size = grid_size
        self.splineorder = spline_order
        self.enable_standalone_scale_spline = stand_alone_spline_scale
        self.base_activation = base_func()
        self.device = device

        size = outdim*indim

        h = (grid_range[1] - grid_range[0]) / grid_size

        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device), torch.linspace(grid_range[0], grid_range[1], steps=grid_size + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)

        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / grid_size
        noises = noises.to(device)
        

        self.spline_weights =  torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, spline_order, device))

        if isinstance(scale_base, float):
            self.c_spl = nn.Parameter(torch.ones(self.indim*self.outdim, device=device) * scale_base).requires_grad_(sb_trainable)
        else:
            self.c_spl = nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        
        if isinstance(scale_sp, float):
            self.c_res = nn.Parameter(torch.ones(self.indim*self.outdim, device=device) * scale_sp).requires_grad_(sp_trainable)
        else:
            self.c_res = nn.Parameter(torch.FloatTensor(scale_sp).to(device)).requires_grad_(sp_trainable)

        if stand_alone_spline_scale:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(outdim, indim)
            )

        #More to do here
        




    def basis(self, x):
        '''
            x - (batch_size, in_features)
            Uses spline_order and current grid to calc values of spline basis functions on input
        '''

        batch = x.shape[0]
        x_e = torch.einsum('ij,k->ikj', x, torch.ones((self.outdim,), device=self.device)).reshape((batch, self.indim*self.outdim))
        x_e = torch.permute(x_e, (1,0))


        bases = spline(x_e, self.grid, self.splineorder, device=self.device)

        return bases.contiguous()
    
    def curve_coeffs(self, x, y):
        """
            Compute coeffs of the curve that interpolate the given points.
            x - Input tensor of shape (batch, in_features)
        """

        A = self.basis(x).permute(0,2,1)
        B = y
        solution = torch.linalg.lstsq( A, B.unsqueeze(dim=2)).solution
        result = solution.permute(2, 0, 1)

        return result.contiguous()
    
    def coeffs_curve(self, x):
        y_eval = torch.einsum('ij,ijk->ik', self.spline_weights, spline(x, self.grid, self.splineorder, device=self.device))
        return y_eval
    
    def update_grid(self, x):

        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = self.coeffs_curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = self.curve_coeffs(x_pos, y_eval)

    #@property
    def scaled_spline_weight(self):
        return self.spline_weights * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x):
        batch = x.shape[0]

        x_e = torch.einsum('ij,k->ikj', x, torch.ones((self.outdim,), device=self.device)).reshape((batch,self.indim*self.outdim))
        x_e = torch.permute(x_e, (1,0))


        res = torch.permute(self.base_activation(x_e), (1,0))

        bi = self.basis(x)
        ci = self.spline_weights

        spline = torch.einsum('ij,ijk->ik', ci, bi)
        spline = torch.permute(spline, (1,0))
        

        const_spline = self.c_spl.unsqueeze(0)
        const_res = self.c_res.unsqueeze(0)

        y = (const_spline * spline) + (const_res * res)

        y_reshape = torch.reshape(y, (batch, self.outdim, self.indim))
        y = (1.0/self.indim)*torch.sum(y_reshape, axis=2)

        grid_reshape = self.grid.reshape(self.outdim, self.indim, -1)
        inp_norm = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-5

        spl_reshape = torch.reshape(spline, (batch, self.outdim, self.indim))

        spl_reg = (torch.mean(torch.abs(spl_reshape), axis=0))/inp_norm

        return y, spl_reg

