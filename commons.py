import math
import torch
from torch import nn
from torch.nn import functional as F


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x

def shift_2d(x):
  b, c, h, w = x.shape
  x = x.view(b, c, h * w)
  x = shift_1d(x)
  x = x.view(b, c, h, w)
  return x


def stft(y, scale='linear'):
  D = torch.stft(y, n_fft=1024, hop_length=256, win_length=1024)
  D = torch.sum(D**2, -1)
  if scale == 'linear':
    return torch.sqrt(D + 1e-10)
  elif scale == 'log':
    S = 0.5 * torch.log(torch.clamp(D, 1e-10, float("inf")))
    return S
  else:
    raise NotImplementedError("Avaliable scaling methods are: linear, log")


def mu_law(x, n_bits=16):
  mu = (2**n_bits - 1)
  x = torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / torch.log(1 + mu)
  return x


def get_same_padding(kernel_size, dilation=1):
  return dilation * (kernel_size // 2)


class DPWrapper(nn.DataParallel):
  """Data Parallel Wrapper"""
  def __getattr__(self, name):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.module, name)