# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from base64 import encode
from math import gamma
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple

from .attention import RelPositionalEncoding
from .conformer import ConformerBlock
from .film import FiLMGen, FiLM
from .utils import make_pad_mask
from .subsampling import Conv2dSubsampling3, LinearNoSubsampling

class Pvad2(nn.Module):
  """
  Conformer: Convolution-augmented Transformer for Speech Recognition
  The paper used a one-lstm Transducer decoder, currently still only implemented
  the conformer encoder shown in the paper.

  Args:
      num_classes (int): Number of classification classes
      input_dim (int, optional): Dimension of input vector
      encoder_dim (int, optional): Dimension of conformer encoder
      num_encoder_layers (int, optional): Number of conformer blocks
      num_attention_heads (int, optional): Number of attention heads
      feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
      conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
      feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
      attention_dropout_p (float, optional): Probability of attention module dropout
      conv_dropout_p (float, optional): Probability of conformer convolution module dropout
      conv_kernel_size (int or tuple, optional): Size of the convolving kernel
      half_step_residual (bool): Flag indication whether to use half step residual or not

  Inputs: inputs
      - **inputs** (batch, time, dim): Tensor containing input vector
      - **input_lengths** (batch): list of sequence input lengths

  Returns: outputs, output_lengths
      - **outputs** (batch, out_channels, time): Tensor produces by conformer.
      - **output_lengths** (batch): list of sequence output lengths
  """
  def __init__(self):
    super(Pvad2, self).__init__()
    # self.fc1 = nn.Linear(512, 64, bias=True)
    self.pos_enc = RelPositionalEncoding(d_model=64, dropout_rate=0.1)

    self.subsample = LinearNoSubsampling(512, 64, dropout_rate=0.1,
                                         pos_enc_class=self.pos_enc)
    self.conformer_block = ConformerBlock(size=64,
                                          attention_heads=8,
                                          linear_units=64,
                                          cnn_module_kernel=7,
                                          causal=True
                                         )

    self.encoder = nn.Sequential(
                              self.conformer_block,
                              self.conformer_block,
                              self.conformer_block,
                              self.conformer_block
                            )
    self.speaker_pre_net = nn.Sequential(
                              self.conformer_block,
                              self.conformer_block
                            )
    self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    # self.cos = F.cosine_similarity(dim=1, eps=1e-6)
    self.gamma_module = nn.Linear(1, 1, bias=True)
    self.beta_module = nn.Linear(1, 1, bias=True)
    self.film_gen = FiLMGen()
    self.film = FiLM()
    self.fc2 = nn.Linear(64, 3, bias=True)
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_normal_(p)

  def count_parameters(self):
    """ Count parameters of encoder """
    return self.encoder.count_parameters()

  @classmethod
  def load_model(cls, path):
    # Load to CPU
    package = torch.load(path, map_location=lambda storage, loc: storage)
    model = cls.load_model_from_package(package)
    return model

  @classmethod
  def load_model_from_package(cls, package):
    model = cls()
    model.load_state_dict(package['state_dict'])
    return model

  @staticmethod
  def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
    package = {
        # state
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if tr_loss is not None:
        package['tr_loss'] = tr_loss
        package['cv_loss'] = cv_loss
    return package

  def update_dropout(self, dropout_p):
    """ Update dropout probability of model """
    self.encoder.update_dropout(dropout_p)

  def forward(self, inputs: Tensor, embedding: Tensor):
    """
    Forward propagate a `inputs` and `targets` pair for training.

    Args:
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        embedding (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.

    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """

    inputs_lens = torch.tensor([x.size(0) for x in inputs], device=inputs.device)

    # subsample
    enc_masks = ~make_pad_mask(inputs_lens, inputs.size(1)).unsqueeze(1)
    enc_pos_inputs, enc_pos_emb, enc_masks = self.subsample(inputs, enc_masks)

    spk_masks = ~make_pad_mask(inputs_lens, inputs.size(1)).unsqueeze(1)
    cur_spk_embd, spk_pos_emb, spk_masks = self.subsample(inputs, spk_masks)

    seq_len = enc_pos_inputs.size(1)
    feat_len = enc_pos_inputs.size(2)
    # encoder
    for layer in self.encoder:
      enc_pos_inputs, enc_masks, _, _ = layer(enc_pos_inputs, enc_masks,
                                              enc_pos_emb)

    # # speaker cosine calculate

    for layer in self.speaker_pre_net:
      cur_spk_embd, spk_masks, _, _ = layer(cur_spk_embd, spk_masks,
                                            spk_pos_emb)

    # cur_spk_embd = cur_spk_embd.view(cur_spk_embd.size(0), cur_spk_embd.size(2),
    #                                  cur_spk_embd.size(1))
    # ref_spk_embd = ref_spk_embd.view(ref_spk_embd.size(0), ref_spk_embd.size(2),
    #                                  ref_spk_embd.size(1))
    embedding = torch.unsqueeze(embedding, 1)
    # print(embedding.size(), cur_spk_embd.size())

    cos_sim = self.cos(cur_spk_embd, embedding) # batch_size * seq_length
    # print("cos_sim", cos_sim, cos_sim.size())
    # print("cur_spk_embd:",cur_spk_embd.size())
    # print("cos_sim:",cos_sim.size())
    # print(cos_sim)
    # FILM
    cos_sim = cos_sim.unsqueeze(2)
    # # print(cos_sim.size())
    # print("cos_sim")
    # print("cos_sim:", cos_sim, cos_sim.size())
    # gamma, beta = self.film_gen(cos_sim)
    gamma = self.gamma_module(cos_sim)
    beta = self.beta_module(cos_sim)
    # print("gamma, beta:", gamma.size(), beta.size())
    # # print(gamma.size(), beta.size())
    # # print(enc_pos_inputs.size())
    # print("enc_pos_inputs", enc_pos_inputs.size())
    scaled_outputs = self.film(enc_pos_inputs, gamma, beta)
    # # print("scaled_outputs", scaled_outputs.size())
    # # fc
    scaled_outputs = scaled_outputs.view(scaled_outputs.size(0), seq_len,
                                         feat_len)
    # print(scaled_outputs.size())
    # scaled_outputs = enc_pos_inputs.view(enc_pos_inputs.size(0), seq_len,
    #                                      feat_len)
    outputs = self.fc2(scaled_outputs)

    return outputs


# if __name__ == "__main__":
#   B = 6
#   L = 12
#   T = 512
#   x = torch.randint(5, (B, L, T), dtype=torch.float32)
#   ref_x = torch.randint(5, (B, L+5, T), dtype=torch.float32)
#   pvad2 = Pvad2()
#   out = pvad2(x, ref_x)
#   print(out.size())

