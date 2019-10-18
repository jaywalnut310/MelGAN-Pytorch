import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons


class Generator(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.pre = nn.Conv1d(in_channels, 512, 7, 1, padding=commons.get_same_padding(7))

    # mid
    self.res_blocks = nn.ModuleList()
    self.res_blocks.append(modules.ResidualBlock(512, 256, 16, 8))
    self.res_blocks.append(modules.ResidualBlock(256, 128, 16, 8))
    self.res_blocks.append(modules.ResidualBlock(128, 64, 4, 2))
    self.res_blocks.append(modules.ResidualBlock(64, 32, 4, 2))

    self.post = nn.Conv1d(32, 1, 7, 1, padding=commons.get_same_padding(7))

    nn.utils.weight_norm(self.pre)
    nn.utils.weight_norm(self.post)

  def forward(self, x):
    # in
    x = self.pre(x)

    # mid
    for l in self.res_blocks:
      x = l(x)

    # out
    x = F.leaky_relu(x)
    x = self.post(x)
    x = torch.tanh(x)
    return x

  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre)
    nn.utils.remove_weight_norm(self.post)
    for l in self.res_blocks:
      l.remove_weight_norm()


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.pre = nn.Conv1d(1, 16, 15, 1, padding=commons.get_same_padding(15))

    # mid
    self.mids = nn.ModuleList()
    self.mids.append(nn.Conv1d(16, 64, 41, 4, groups=4, padding=commons.get_same_padding(41)))
    self.mids.append(nn.Conv1d(64, 256, 41, 4, groups=16, padding=commons.get_same_padding(41)))
    self.mids.append(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=commons.get_same_padding(41)))
    self.mids.append(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=commons.get_same_padding(41)))
    self.mids.append(nn.Conv1d(1024, 1024, 5, 1, padding=commons.get_same_padding(5)))

    self.post = nn.Conv1d(1024, 1, 3, 1, padding=commons.get_same_padding(3))

    nn.utils.weight_norm(self.pre)
    for l in self.mids:
      nn.utils.weight_norm(l)
    nn.utils.weight_norm(self.post)

  def forward(self, x, return_many=False):
    rets = []
    # in
    x = self.pre(x)
    x = F.leaky_relu(x)
    rets.append(x)

    # mid
    for l in self.mids:
      x = l(x)
      x = F.leaky_relu(x)
      rets.append(x)

    # out
    x = self.post(x)
    rets.append(x)

    if return_many:
      return x, rets
    else:
      return x

  def remove_weight_norm(self):
    nn.utils.remove_weight_norm(self.pre)
    for l in self.mids:
      nn.utils.remove_weight_norm(l)
    nn.utils.remove_weight_norm(self.post)


class MultiScaleDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.discs = nn.ModuleList()
    for _ in range(3):
      self.discs.append(Discriminator())

    self.poolings = nn.ModuleList()
    self.poolings.append(nn.AvgPool1d(4, 2, commons.get_same_padding(4)))
    self.poolings.append(nn.AvgPool1d(4, 4, commons.get_same_padding(4)))

  def forward(self, x, return_many=False):
    ys = []
    rets = []
    for i, l in enumerate(self.discs):
      if i > 0:
        x = self.poolings[i-1](x)
      y, ret = l(x, return_many=True)
      ys.append(y)
      rets.extend(ret)

    if return_many:
      return ys, rets
    else:
      return ys

  def remove_weight_norm(self):
    for l in self.discs:
      l.remove_weight_norm()