import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class Ladd(Function):
    @staticmethod
    def forward(ctxt, x, pat, cout, cin, kernel_size, padding, stride):
        """
        The actual formula is:
            mag * (1 - mean(x - pat)) + bias
        But the `mean` op is simply the sum div by `kernel_size^2`, so we let
        the constant coe be absorbed by `mag`. (This is what I expected to do if
        I'm about to implement it in CUDA.) Same applies to the bias term
        absorbing `bias * mag * 1`. Then we have:
            mag * sum(x - pat) + bias
        """
        n, _, h, w = x.size()
        npoint = kernel_size * kernel_size
        hout = (h - kernel_size + 2 * padding) // stride + 1
        wout = (w - kernel_size + 2 * padding) // stride + 1
        npatch = hout * wout
        npatch_point = cin * npoint
        # Tiled input (n, cin, kernel_size^2).
        x_tiles = F.unfold(x.reshape(n * cin, 1, h, w),
                           kernel_size,
                           padding=padding,
                           stride=stride)
        # `x_tiles` at this point is (n * cin, npoint, npatch).
        x_tiles = x_tiles.reshape((n, cin, npoint, npatch))
        x_tiles = x_tiles.permute(0, 3, 1, 2)
        x_tiles = x_tiles.reshape((n * npatch, npatch_point))
        # `pat` at this point is (cout, cin, npoint).
        pat = pat.reshape(cout, npatch_point)

        x_tiles = x_tiles.unsqueeze(1).repeat(1, cout, 1)
        pat = pat.unsqueeze(0).repeat(n * npatch, 1, 1)

        # (n * npatch, cout, npatch_point)
        minus_delta = pat - x_tiles
        minus_delta_mean = minus_delta.abs().sum(dim=2)
        y = - minus_delta_mean

        minus_delta = minus_delta.mean(axis=0).reshape(cout, cin, npoint)
        ctxt.save_for_backward(minus_delta)
        ctxt.kernel_size = kernel_size
        ctxt.padding = padding
        ctxt.stride = stride

        #print("---", pat.min(), pat.mean(), pat.max())

        return y.contiguous().view(n, hout, wout, cout).permute(0,3,1,2)
    @staticmethod
    def backward(ctxt, grad_output):
        minus_delta, = ctxt.saved_tensors
        minus_delta = minus_delta
        n, cout, hout, wout = grad_output.size()
        _, cin, npoint = npatch_point = minus_delta.size()
        npatch_point = cin * npoint

        dy_tiles = F.unfold(grad_output.reshape(n * cout, 1, hout, wout),
                            ctxt.kernel_size,
                            padding=ctxt.padding,
                            stride=ctxt.stride)
        # `dy_tiles` at this point is (n * cout, npoint, npatch).
        _, npoint, npatch = dy_tiles.size()
        dy_tiles = dy_tiles.reshape(n, cout, npoint, npatch).sum(axis=0).sum(axis=2)
        dpat = minus_delta * (-dy_tiles.unsqueeze(1).clamp(-1, 1))
        ita = 1.0 # This is the adder filter hyper-param but aint gonna use it.
        minus_delta_mag = ((minus_delta * minus_delta).sum(axis=(1, 2), keepdim=True)).sqrt()
        local_lr_coe = ita * np.sqrt(npatch_point) / minus_delta_mag
        #print("///", dpat.min(), dpat.mean(), dpat.max())

        return (None, dpat * local_lr_coe, None, None, None, None, None)


class Ladder2D(nn.Module):
    def __init__(self, nchannel_in, nchannel_out, kernel_size,
                 stride=1, padding=0):
        super(Ladder2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.nchannel_in = nchannel_in
        self.nchannel_out = nchannel_out
        self.kernel_size = kernel_size
        # Pattern.
        pat_shape = (nchannel_out, nchannel_in, kernel_size * kernel_size)
        self.pat = Parameter(nn.init.uniform_(torch.randn(*pat_shape)))

    def forward(self, x):
        return Ladd.apply(x, self.pat, self.nchannel_out, self.nchannel_in,
                          self.kernel_size, self.padding, self.stride)
