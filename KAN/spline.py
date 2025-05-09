import torch
import numpy as np

def spline(x, grid, k=0, extend=True, device='cuda'):
    '''
    eval x on b-spline bases calculated from grid
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = spline(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value

def coef2curve(x_e, grid, coef, k, device='cuda'):


    if coef.dtype != x_e.dtype:
        coef = coef.to(x_e.dtype)
    y_e = torch.einsum('ij,ijk->ik', coef, spline(x_e, grid, k, device=device))

    return y_e

def curve2coef(x_e, y_e, grid, k, device='cuda'):
    mat = spline(x_e, grid, k ,device=device).permute(0,2,1)

    coef = torch.linalg.lstsq(mat.to(device), y_e.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    
    return coef.to(device)
