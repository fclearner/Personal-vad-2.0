"""Subsampling layer definition."""

from typing import Tuple

import torch


class BaseSubsampling(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.right_context = 0
    self.subsampling_rate = 1

  def position_encoding(self, offset: int, size: int):
    return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
  """Linear transform the input without subsampling

  Args:
    idim (int): Input dimension.
    odim (int): Output dimension.
    dropout_rate (float): Dropout rate.

  """
  def __init__(self, idim: int, odim: int, dropout_rate: float,
         pos_enc_class: torch.nn.Module):
    """Construct an linear object."""
    super().__init__()
    self.out = torch.nn.Sequential(
      torch.nn.Linear(idim, odim),
      torch.nn.LayerNorm(odim, eps=1e-5),
      torch.nn.Dropout(dropout_rate),
    )
    self.pos_enc = pos_enc_class
    self.right_context = 0
    self.subsampling_rate = 1

  def forward(
      self,
      x: torch.Tensor,
      x_mask: torch.Tensor,
      offset: int = 0
  ):
    """Input x.

    Args:
      x (torch.Tensor): Input tensor (#batch, time, idim).
      x_mask (torch.Tensor): Input mask (#batch, 1, time).

    Returns:
      torch.Tensor: linear input tensor (#batch, time', odim),
        where time' = time .
      torch.Tensor: linear input mask (#batch, 1, time'),
        where time' = time .

    """
    x = self.out(x)
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask


class Conv2dSubsampling3(BaseSubsampling):
  """Convolutional 2D subsampling (to 1/3 length).
  Args:
    idim (int): Input dimension.
    odim (int): Output dimension.
    dropout_rate (float): Dropout rate.
  """
  def __init__(self, idim: int, odim: int, dropout_rate: float,
               pos_enc_class: torch.nn.Module):
    """Construct an Conv2dSubsampling8 object."""
    super().__init__()
    self.conv = torch.nn.Sequential(
      torch.nn.Conv2d(1, odim, 5, 3),
      torch.nn.ReLU(),
    )
    self.linear = torch.nn.Linear(
      odim * ((idim - 2) // 3), odim)
    self.pos_enc = pos_enc_class
    self.subsampling_rate = 3
    # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
    # 4 = (5 - 1) * 1
    self.right_context = 4

  def forward(
      self,
      x: torch.Tensor,
      x_mask: torch.Tensor,
      offset: int = 0
    ):
    """ Subsample x.
    Args:
      x (torch.Tensor): Input tensor (#batch, time, idim).
      x_mask (torch.Tensor): Input mask (#batch, 1, time).
    Returns:
      torch.Tensor: Subsampled tensor (#batch, time', odim),
        where time' = time // 8.
      torch.Tensor: Subsampled mask (#batch, 1, time'),
        where time' = time // 8.
      torch.Tensor: positional encoding
    """
    x = x.unsqueeze(1)  # (b, c, t, f)
    # for layer in self.conv:
    #   x = layer(x)
    x = self.conv(x)
    b, c, t, f = x.size()

    x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask[:, :, :-4:3]
