# from tkinter import X
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform_


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = kaiming_normal
  elif init.lower() == 'uniform':
    init_params = kaiming_uniform_
  else:
    return
  for m in modules:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      init_params(m.weight)


def init_rnn(rnn_type, hidden_dim1, hidden_dim2, rnn_num_layers,
             dropout=0, bidirectional=False):
  if rnn_type == 'gru':
    return nn.GRU(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                  batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'lstm':
    return nn.LSTM(hidden_dim1, hidden_dim2, rnn_num_layers, dropout=dropout,
                   batch_first=True, bidirectional=bidirectional)
  elif rnn_type == 'linear':
    return None
  else:
    print('RNN type ' + str(rnn_type) + ' not yet implemented.')
    raise(NotImplementedError)


class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    # gammas = gammas.view(x.size(0), x.size(1), 1)
    # betas = betas.view(x.size(0), x.size(1), 1)
    return (gammas * x) + betas


class FiLMGen(nn.Module):
  def __init__(self,
    input_dim=1,
    hidden_dim=4,
    rnn_num_layers=1,
    rnn_dropout=0,
    output_batchnorm=False,
    bidirectional=False,
    encoder_type='gru',
    decoder_type='gru',
    num_modules=1,
    module_num_layers=1,
    module_dim=1,
  ):
    super(FiLMGen, self).__init__()
    # self.input_dim = input_dim
    self.encoder_type = encoder_type
    self.decoder_type = decoder_type
    self.output_batchnorm = output_batchnorm
    self.bidirectional = bidirectional
    self.num_dir = 1 if not self.bidirectional else 2
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim
    if self.bidirectional:
      if decoder_type != 'linear':
        raise(NotImplementedError)
      hidden_dim = (int) (hidden_dim / self.num_dir)

    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock
    self.encoder_rnn = init_rnn(self.encoder_type, input_dim, hidden_dim,
                                rnn_num_layers, dropout=rnn_dropout,
                                bidirectional=self.bidirectional)
    self.decoder_rnn = init_rnn(self.decoder_type, hidden_dim, hidden_dim,
                                rnn_num_layers, dropout=rnn_dropout,
                                bidirectional=self.bidirectional)

    self.decoder_linear = nn.Linear(
      hidden_dim * self.num_dir, self.num_modules * self.cond_feat_size)
    if self.output_batchnorm:
      self.output_bn = nn.BatchNorm1d(self.cond_feat_size, affine=True)

    init_modules(self.modules())

  def get_dims(self, x=None):
    H = self.encoder_rnn.hidden_size
    L = self.encoder_rnn.num_layers

    N = x.size(0) if x is not None else None
    return H, L, N

  def encoder(self, x):
    H, L, N = self.get_dims(x=x)

    h0 = Variable(torch.zeros(L, N, H).type_as(x.data)) # num_layers*batch_size*hidden_size

    if self.encoder_type == 'lstm':
      c0 = Variable(torch.zeros(L, N, H).type_as(x.data))
      out, _ = self.encoder_rnn(x, (h0, c0))
    elif self.encoder_type == 'gru':
      out, _ = self.encoder_rnn(x, h0)

    return out

  def decoder(self, encoded, h0=None, c0=None):
    H, L, N = self.get_dims(x=encoded)

    T_out = self.num_modules
    V_out = self.cond_feat_size

    if self.decoder_type == 'linear':
      output_shaped = self.decoder_linear(encoded)
      gammas, betas = torch.split(output_shaped[:,:,:2*self.module_dim],
                                  self.module_dim, dim=-1)
      return gammas, betas

    encoded_repeat = encoded.expand(N, T_out*encoded.size(1), H)
    if not h0:
      h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))

    if self.decoder_type == 'lstm':
      if not c0:
        c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
      rnn_output, (_, _) = self.decoder_rnn(encoded_repeat, (h0, c0))
    elif self.decoder_type == 'gru':
      rnn_output, _ = self.decoder_rnn(encoded_repeat, h0)

    rnn_output_2d = rnn_output.contiguous().view(N, rnn_output.size(1) * T_out, H)
    linear_output = self.decoder_linear(rnn_output_2d)

    if self.output_batchnorm:
      linear_output = self.output_bn(linear_output)
    output_shaped = linear_output.view(N, rnn_output.size(1) * T_out, V_out)

    gammas, betas = torch.split(output_shaped[:,:,:2*self.module_dim],
                                self.module_dim, dim=-1)

    return gammas, betas

  def forward(self, x):
    self.encoder_rnn.flatten_parameters()
    encoded = self.encoder(x)
    gammas, betas = self.decoder(encoded)

    return gammas, betas


# if __name__ == "__main__":
#   import numpy as np
#   B = 8
#   x = torch.randint(5, (B, 440, 64))
#   cos = torch.rand((B, 440))
#   fim_gen = FiLMGen()
#   gammas, betas = fim_gen(cos)
#   # gammas = torch.tensor([0.1*i*np.ones((8,1)) for i in range(8)])
#   # betas = 0*torch.ones((B,8,1))
#   # print(gammas, betas)
#   film = FiLM()
#   y = film(x, gammas, betas)