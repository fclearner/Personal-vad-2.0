from typing import Optional, Tuple

import math
import torch
from torch import nn
from .utils import get_activation
from .attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .position_wise_feed_forward import PositionwiseFeedForward
from .convolution import ConvolutionModule
from typeguard import check_argument_types


class ConformerBlock(nn.Module):
  """Encoder layer module from wenet.
    Args:
      size (int): Input dimension.
      self_attn (torch.nn.Module): Self-attention module instance.
          `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
          instance can be used as the argument.
      feed_forward (torch.nn.Module): Feed-forward module instance.
          `PositionwiseFeedForward` instance can be used as the argument.
      feed_forward_macaron (torch.nn.Module): Additional feed-forward module
            instance.
          `PositionwiseFeedForward` instance can be used as the argument.
      conv_module (torch.nn.Module): Convolution module instance.
          `ConvlutionModule` instance can be used as the argument.
      dropout_rate (float): Dropout rate.
      normalize_before (bool):
          True: use layer_norm before each sub-block.
          False: use layer_norm after each sub-block.
      concat_after (bool): Whether to concat attention layer's input and
          output.
          True: x -> x + linear(concat(x, att(x)))
          False: x -> x + att(x)
  """
  def __init__(
    self,
    size: int = 512,
    attention_heads: int = 4,
    linear_units: int = 2048,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.0,
    pos_enc_layer_type: str = "rel_pos",
    normalize_before: bool = True,
    concat_after: bool = False,
    macaron_style: bool = True,
    activation_type: str = "swish",
    cnn_module_kernel: int = 31,
    causal: bool = False,
    cnn_module_norm: str = "batch_norm"
  ):
    """Construct an EncoderLayer object."""
    assert check_argument_types()
    super(ConformerBlock, self).__init__()
    activation = get_activation(activation_type)
    if pos_enc_layer_type == "no_pos":
      self.self_attn = MultiHeadedAttention(attention_heads, size,
                                            attention_dropout_rate)
    else:
      self.self_attn = RelPositionMultiHeadedAttention(attention_heads, size,
                                                       attention_dropout_rate)

    self.feed_forward = PositionwiseFeedForward(size, linear_units,
                                                dropout_rate, activation)
    if macaron_style:
        self.feed_forward_macaron = PositionwiseFeedForward(size,
                                                            linear_units,
                                                            dropout_rate,
                                                            activation)
    self.conv_module = ConvolutionModule(size, cnn_module_kernel,
                                         activation, cnn_module_norm, causal)
    self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
    self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
    if self.feed_forward_macaron is not None:
      self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
      self.ff_scale = 0.5
    else:
      self.ff_scale = 1.0
    if self.conv_module is not None:
      self.norm_conv = nn.LayerNorm(size,
                                    eps=1e-5)  # for the CNN module
      self.norm_final = nn.LayerNorm(size, eps=1e-5)  # for the final output of the block
    self.dropout = nn.Dropout(dropout_rate)
    self.size = size
    self.normalize_before = normalize_before
    self.concat_after = concat_after
    self.concat_linear = nn.Linear(size + size, size)

  def forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor,
    pos_emb: torch.Tensor,
    mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
  ):
    """Compute encoded features.

    Args:
        x (torch.Tensor): (#batch, time, size)
        mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
            (0, 0, 0) means fake mask.
        pos_emb (torch.Tensor): positional encoding, must not be None
            for ConformerEncoderLayer.
        mask_pad (torch.Tensor): batch padding mask used for conv module.
            (#batch, 1，time), (0, 0, 0) means fake mask.
        att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
            (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
        cnn_cache (torch.Tensor): Convolution cache in conformer layer
            (#batch=1, size, cache_t2)
    Returns:
        torch.Tensor: Output tensor (#batch, time, size).
        torch.Tensor: Mask tensor (#batch, time, time).
        torch.Tensor: att_cache tensor,
            (#batch=1, head, cache_t1 + time, d_k * 2).
        torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
    """

    # whether to use macaron style
    if self.feed_forward_macaron is not None:
      residual = x
      if self.normalize_before:
        x = self.norm_ff_macaron(x)
      x = residual + self.ff_scale * self.dropout(
        self.feed_forward_macaron(x))
      if not self.normalize_before:
        x = self.norm_ff_macaron(x)

    # multi-headed self-attention module
    residual = x
    if self.normalize_before:
      x = self.norm_mha(x)

    x_att, new_att_cache = self.self_attn(
      x, x, x, mask, pos_emb, att_cache)
    if self.concat_after:
      x_concat = torch.cat((x, x_att), dim=-1)
      x = residual + self.concat_linear(x_concat)
    else:
      x = residual + self.dropout(x_att)
    if not self.normalize_before:
      x = self.norm_mha(x)

    # convolution module
    # Fake new cnn cache here, and then change it in conv_module
    new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
    if self.conv_module is not None:
      residual = x
      if self.normalize_before:
        x = self.norm_conv(x)
      x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
      x = residual + self.dropout(x)

      if not self.normalize_before:
        x = self.norm_conv(x)

    # feed forward module
    residual = x
    if self.normalize_before:
      x = self.norm_ff(x)

    x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
    if not self.normalize_before:
      x = self.norm_ff(x)

    if self.conv_module is not None:
      x = self.norm_final(x)

    return x, mask, new_att_cache, new_cnn_cache
