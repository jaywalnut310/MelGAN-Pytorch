import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons


class Generator(nn.Module):
  def __init__(self, in_channels, noise=False):
    super().__init__()
    self.in_channels = in_channels
    self.noise = noise

    self.pre = nn.Conv1d(in_channels, 512, 7, 1, padding=commons.get_same_padding(7))

    # mid
    self.res_blocks = nn.ModuleList()
    self.res_blocks.append(modules.ResidualBlock(512, 256, 16, 8, padding=4))
    self.res_blocks.append(modules.ResidualBlock(256, 128, 16, 8, padding=4))
    self.res_blocks.append(modules.ResidualBlock(128, 64, 4, 2, padding=1))
    self.res_blocks.append(modules.ResidualBlock(64, 32, 4, 2, padding=1))

    self.post = nn.Conv1d(32, 1, 7, 1, padding=commons.get_same_padding(7))

    if noise:
      self.noise_dim = 128
      self.noise_ms = nn.ModuleList()
      self.noise_ms.append(nn.Linear(self.noise_dim, 256))
      self.noise_ms.append(nn.Linear(self.noise_dim, 128))
      self.noise_ms.append(nn.Linear(self.noise_dim, 64))
      self.noise_ms.append(nn.Linear(self.noise_dim, 32))
      self.noise_ss = nn.ModuleList()
      self.noise_ss.append(nn.Linear(self.noise_dim, 256))
      self.noise_ss.append(nn.Linear(self.noise_dim, 128))
      self.noise_ss.append(nn.Linear(self.noise_dim, 64))
      self.noise_ss.append(nn.Linear(self.noise_dim, 32))

    nn.utils.weight_norm(self.pre)
    nn.utils.weight_norm(self.post)

  def forward(self, x):
    # in
    x = self.pre(x)
    if self.noise:
      z = torch.randn(x.size(0), self.noise_dim)
      z = z.to(x.dtype).to(x.device)

    # mid
    for i, l in enumerate(self.res_blocks):
      x = l(x)
      if self.noise:
        m = self.noise_ms[i](z).unsqueeze(-1)
        logs = self.noise_ss[i](z).unsqueeze(-1)
        x = torch.exp(logs) * x + m
    
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

  def forward(self, x, c=None, return_many=False):
    rets = []
    # in
    x = self.pre(x)
    x = F.leaky_relu(x)
    rets.append(x)

    # mid
    for l in self.mids[:-1]:
      x = l(x)
      x = F.leaky_relu(x)
      rets.append(x)
    if c is not None:
      x = x + c
    x = self.mids[-1](x)
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
  def __init__(self, condition_type="u", condition_channels=80):
    super().__init__()
    self.condition_type = condition_type
    self.condition_channels = condition_channels

    self.discs = nn.ModuleList()
    for _ in range(3):
      self.discs.append(Discriminator())

    self.poolings = nn.ModuleList()
    self.poolings.append(nn.AvgPool1d(4, 2, 1))
    self.poolings.append(nn.AvgPool1d(4, 4, 0))

    if condition_type=="u": # unconditional
      pass
    elif condition_type in ["c", "b"]: # conditional
      self.cond_convs = nn.ModuleList()
      self.cond_convs.append(nn.Conv1d(condition_channels, 1024, 1, 1))
      self.cond_convs.append(nn.Conv1d(1024, 1024, 4, 2, 1))
      self.cond_convs.append(nn.Conv1d(1024, 1024, 4, 2, 1))
      if condition_type=="b": # both
        self.discs_ = nn.ModuleList()
        for _ in range(3):
          self.discs_.append(Discriminator())
    else:
      raise NotImplementedError("Available condition_types are: u, c and b.")
    

  def forward(self, x, c=None, return_many=False):
    x_org = x
    ys = []
    rets = []
    for i, l in enumerate(self.discs):
      if i > 0:
        x = self.poolings[i-1](x)
      if c is not None and self.condition_type in ["c", "b"]:
        c = self.cond_convs[i](c)
      y, ret = l(x, c, return_many=True)
      ys.append(y)
      rets.extend(ret)

    if self.condition_type == "b":
      x = x_org
      for i, l in enumerate(self.discs_):
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
    if self.condition_type == "b":
      for l in self.discs_:
        l.remove_weight_norm()
