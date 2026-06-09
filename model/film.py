import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, kaiming_uniform_


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = kaiming_normal_
  elif init.lower() == 'uniform':
    init_params = kaiming_uniform_
  else:
    return

  for module in modules:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
      init_params(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_rnn(rnn_type, input_dim, hidden_dim, num_layers, dropout=0,
             bidirectional=False):
  rnn_type = rnn_type.lower()
  if rnn_type == 'gru':
    return nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout,
                  batch_first=True, bidirectional=bidirectional)
  if rnn_type == 'lstm':
    return nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout,
                   batch_first=True, bidirectional=bidirectional)
  if rnn_type == 'linear':
    return None
  raise NotImplementedError(f'RNN type {rnn_type} is not implemented.')


class FiLM(nn.Module):
  """Feature-wise affine modulation: y = gamma * x + beta."""

  def forward(self, x, gammas, betas):
    if gammas.dim() == 2:
      gammas = gammas.unsqueeze(1)
    if betas.dim() == 2:
      betas = betas.unsqueeze(1)
    return (gammas * x) + betas


class FiLMGen(nn.Module):
  """Generate FiLM parameters from a conditioning sequence.

  Input shape is ``(batch, time, input_dim)``. The default configuration returns
  ``gammas`` and ``betas`` with shape ``(batch, time, module_dim)``.
  """

  def __init__(
      self,
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
    self.encoder_type = encoder_type.lower()
    self.decoder_type = decoder_type.lower()
    self.output_batchnorm = output_batchnorm
    self.bidirectional = bidirectional
    self.num_dir = 1 if not self.bidirectional else 2
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim
    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers

    if self.bidirectional and hidden_dim % self.num_dir != 0:
      raise ValueError('hidden_dim must be divisible by 2 when bidirectional.')

    rnn_hidden_dim = hidden_dim // self.num_dir if self.bidirectional else hidden_dim
    self.encoder_rnn = init_rnn(self.encoder_type, input_dim, rnn_hidden_dim,
                                rnn_num_layers, dropout=rnn_dropout,
                                bidirectional=self.bidirectional)
    decoder_input_dim = rnn_hidden_dim * self.num_dir
    self.decoder_rnn = init_rnn(self.decoder_type, decoder_input_dim,
                                rnn_hidden_dim, rnn_num_layers,
                                dropout=rnn_dropout,
                                bidirectional=self.bidirectional)
    decoder_output_dim = (
        decoder_input_dim if self.decoder_type == 'linear'
        else rnn_hidden_dim * self.num_dir)
    self.decoder_linear = nn.Linear(
        decoder_output_dim, self.num_modules * self.cond_feat_size)

    if self.output_batchnorm:
      self.output_bn = nn.BatchNorm1d(
          self.num_modules * self.cond_feat_size, affine=True)

    init_modules(self.modules())

  @staticmethod
  def _initial_state(rnn, x):
    num_layers = rnn.num_layers * (2 if rnn.bidirectional else 1)
    state = x.new_zeros(num_layers, x.size(0), rnn.hidden_size)
    if isinstance(rnn, nn.LSTM):
      return state, state.clone()
    return state

  def encoder(self, x):
    if self.encoder_rnn is None:
      return x
    return self.encoder_rnn(x, self._initial_state(self.encoder_rnn, x))[0]

  def decoder(self, encoded):
    if self.decoder_rnn is None:
      decoded = encoded
    else:
      decoded = self.decoder_rnn(
          encoded, self._initial_state(self.decoder_rnn, encoded))[0]

    linear_output = self.decoder_linear(decoded)
    if self.output_batchnorm:
      bsz, steps, dim = linear_output.size()
      linear_output = self.output_bn(linear_output.reshape(bsz * steps, dim))
      linear_output = linear_output.view(bsz, steps, dim)

    bsz, steps, _ = linear_output.size()
    shaped = linear_output.view(
        bsz, steps, self.num_modules, self.module_num_layers,
        2, self.module_dim)
    gammas = shaped[..., 0, :]
    betas = shaped[..., 1, :]

    if self.num_modules == 1 and self.module_num_layers == 1:
      gammas = gammas.squeeze(2).squeeze(2)
      betas = betas.squeeze(2).squeeze(2)

    return gammas, betas

  def forward(self, x):
    if x.dim() == 2:
      x = x.unsqueeze(-1)
    if self.encoder_rnn is not None:
      self.encoder_rnn.flatten_parameters()
    if self.decoder_rnn is not None:
      self.decoder_rnn.flatten_parameters()

    encoded = self.encoder(x)
    return self.decoder(encoded)
