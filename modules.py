import math
import torch
from torch import nn
from torch.nn import functional as F

import commons


class ResidualStack(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels

    self.res_1 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, padding=commons.get_same_padding(3)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, padding=commons.get_same_padding(3))
    )
    self.res_2 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, dilation=3, padding=commons.get_same_padding(3, 3)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, padding=commons.get_same_padding(3))
    )
    self.res_3 = nn.Sequential(
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, dilation=9, padding=commons.get_same_padding(3, 9)),
      nn.LeakyReLU(),
      nn.Conv1d(channels, channels, 3, padding=commons.get_same_padding(3))
    )

    nn.utils.weight_norm(self.res_1[1])
    nn.utils.weight_norm(self.res_1[3])
    nn.utils.weight_norm(self.res_2[1])
    nn.utils.weight_norm(self.res_2[3])
    nn.utils.weight_norm(self.res_3[1])
    nn.utils.weight_norm(self.res_3[3])

  def forward(self, x):
    for l in [self.res_1, self.res_2, self.res_3]:
      x_ = l(x)
      x = x + x_
    return x
  
  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.res_1[1])
    nn.utils.remove_weight_norm(self.res_1[3])
    nn.utils.remove_weight_norm(self.res_2[1])
    nn.utils.remove_weight_norm(self.res_2[3])
    nn.utils.remove_weight_norm(self.res_3[1])
    nn.utils.remove_weight_norm(self.res_3[3])
  

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.pre = nn.Sequential(
      nn.LeakyReLU(),
      nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=padding)
    )
    self.res_stack = ResidualStack(out_channels)

    nn.utils.weight_norm(self.pre[1])

  def forward(self, x):
    x = self.pre(x)
    x = self.res_stack(x)
    return x

  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre[1])
    self.res_stack.remove_weight_norm()