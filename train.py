import os
import json
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import LJspeechDataset, collate_fn, collate_fn_synthesize

import models
import commons
import utils
                            
                            
hps = utils.get_hparams()
logger = utils.get_logger(hps.model_dir)
logger.info(hps)
utils.check_git_hash(hps.model_dir)
writer = SummaryWriter(log_dir=hps.model_dir)
global_step = 0

use_cuda = hps.train.use_cuda and torch.cuda.is_available()
torch.manual_seed(hps.train.seed)
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
train_dataset = LJspeechDataset(hps.data.data_path, True, 0.1)
test_dataset = LJspeechDataset(hps.data.data_path, False, 0.1)
train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size, shuffle=True, collate_fn=collate_fn,
                          **kwargs)
test_loader = DataLoader(test_dataset, batch_size=hps.train.batch_size, collate_fn=collate_fn,
                          **kwargs)

generator = models.Generator(hps.data.n_channels, noise=hps.model.noise).to(device)
discriminator = models.MultiScaleDiscriminator(hps.model.condition_type, hps.data.n_channels).to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas)


def feature_matching_loss(rs_t, rs_f):
  l_tot = 0
  for d_t, d_f in zip(rs_t, rs_f):
    l_tot += torch.mean(torch.abs(d_t - d_f))
  return l_tot


def train(epoch):
  global global_step

  generator.train()
  discriminator.train()
  for batch_idx, (x, c, _) in enumerate(train_loader):
    x, c = x.to(device), c.to(device)
    if hps.data.mu_law:
      x = commons.mu_law(x)
    if hps.model.condition_type == "u":
      c_cond = None
    else:
      c_cond = c

    #  Train Discriminator
    x_gen = generator(c).detach()
    for _ in range(hps.train.n_iter):
      optimizer_d.zero_grad()

      ys_t, rets_t = discriminator(x, c_cond, return_many=True)
      ys_f, rets_f = discriminator(x_gen, c_cond, return_many=True)

      loss_ds_t = []
      loss_ds_f = []
      for y_t, y_f in zip(ys_t, ys_f):
        loss_ds_t.append(torch.mean(torch.sum((y_t - 1)**2, [1, 2])))
        loss_ds_f.append(torch.mean(torch.sum((y_f)**2, [1, 2])))
        # loss_ds_t.append(torch.mean((y_t - 1)**2))
        # loss_ds_f.append(torch.mean((y_f)**2))
      loss_d = sum(loss_ds_t) + sum(loss_ds_f)

      loss_d.backward()
      optimizer_d.step()

    # Train Generator
    optimizer_g.zero_grad()
    
    x_gen = generator(c)
    ys_t, rets_t = discriminator(x, c_cond, return_many=True)
    ys_f, rets_f = discriminator(x_gen, c_cond, return_many=True)

    loss_gs = []
    for y in ys_f:
      loss_gs.append(torch.mean(torch.sum((y - 1)**2, [1, 2])))
      # loss_gs.append(torch.mean((y - 1)**2))
    loss_fm = feature_matching_loss(rets_t, rets_f)
    loss_g = sum(loss_gs) + hps.train.c_fm * loss_fm

    loss_g.backward()
    optimizer_g.step()

    if batch_idx % hps.train.log_interval == 0:
      logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} / {:.6f}'.format(
        epoch, batch_idx * len(x), len(train_loader.dataset),
        100. * batch_idx / len(train_loader),
        loss_d.item(), loss_g.item()))
      logger.info([x.item() for x in loss_ds_t + loss_ds_f + loss_gs + [loss_fm]] + [global_step])
      
      if hps.data.mu_law:
        x = commons.mu_law(x, inversion=True)
        x_gen = commons.mu_law(x_gen, inversion=True)
      scalar_dict = {"loss/d/total": loss_d, "loss/g/total": loss_g, "loss/g/fm": loss_fm}
      scalar_dict.update({"loss/d/t_{}".format(i): v for i, v in enumerate(loss_ds_t)})
      scalar_dict.update({"loss/d/f_{}".format(i): v for i, v in enumerate(loss_ds_f)})
      scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
      utils.summarize(
        writer=writer,
        global_step=global_step, 
        images={"x_org": utils.plot_spectrogram_to_numpy(commons.stft(x[:1,0])[0].data.cpu().numpy()), 
          "x_gen": utils.plot_spectrogram_to_numpy(commons.stft(x_gen[:1,0])[0].data.cpu().numpy())}, 
        scalars=scalar_dict)
    global_step += 1
  logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
  for epoch in range(1, hps.train.epochs + 1):
    train(epoch)
    utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    utils.save_checkpoint(discriminator, optimizer_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(epoch)))

                            
