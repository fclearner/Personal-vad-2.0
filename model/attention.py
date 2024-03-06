import math
import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
  """Multi-Head Attention layer.

  Args:
    n_head (int): The number of heads.
    n_feat (int): The number of features.
    dropout_rate (float): Dropout rate.

  """
  def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
    """Construct an MultiHeadedAttention object."""
    super().__init__()
    assert n_feat % n_head == 0
    # We assume d_v always equals d_k
    self.d_k = n_feat // n_head
    self.h = n_head
    self.linear_q = nn.Linear(n_feat, n_feat)
    self.linear_k = nn.Linear(n_feat, n_feat)
    self.linear_v = nn.Linear(n_feat, n_feat)
    self.linear_out = nn.Linear(n_feat, n_feat)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward_qkv(
    self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ):
    """Transform query, key and value.
  
    Args:
      query (torch.Tensor): Query tensor (#batch, time1, size).
      key (torch.Tensor): Key tensor (#batch, time2, size).
      value (torch.Tensor): Value tensor (#batch, time2, size).

    Returns:
      torch.Tensor: Transformed query tensor, size
        (#batch, n_head, time1, d_k).
      torch.Tensor: Transformed key tensor, size
          (#batch, n_head, time2, d_k).
      torch.Tensor: Transformed value tensor, size
        (#batch, n_head, time2, d_k).

    """

    n_batch = query.size(0)
    q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
    k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
    v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
    q = q.transpose(1, 2)  # (batch, head, time1, d_k)
    k = k.transpose(1, 2)  # (batch, head, time2, d_k)
    v = v.transpose(1, 2)  # (batch, head, time2, d_k)

    return q, k, v

  def forward_attention(
    self, value: torch.Tensor, scores: torch.Tensor,
    mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
  ):
    """Compute attention context vector.

    Args:
        value (torch.Tensor): Transformed value, size
            (#batch, n_head, time2, d_k).
        scores (torch.Tensor): Attention score, size
            (#batch, n_head, time1, time2).
        mask (torch.Tensor): Mask, size (#batch, 1, time2) or
            (#batch, time1, time2), (0, 0, 0) means fake mask.

    Returns:
        torch.Tensor: Transformed value (#batch, time1, d_model)
            weighted by the attention score (#batch, time1, time2).

    """
    n_batch = value.size(0)
    if mask.size(2) > 0 :  # time2 > 0
        mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
        scores = scores.masked_fill(mask, -float('inf'))
        attn = torch.softmax(scores, dim=-1).masked_fill(
            mask, 0.0)  # (batch, head, time1, time2)
    else:
        attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

    p_attn = self.dropout(attn)

    x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
    x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                self.h * self.d_k)
            )  # (batch, time1, d_model)

    return self.linear_out(x)  # (batch, time1, d_model)

  def forward(self, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
              pos_emb: torch.Tensor = torch.empty(0),
              cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
             ):
    """Compute scaled dot product attention.

    Args:
        query (torch.Tensor): Query tensor (#batch, time1, size).
        key (torch.Tensor): Key tensor (#batch, time2, size).
        value (torch.Tensor): Value tensor (#batch, time2, size).
        mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
            (#batch, time1, time2).
            1.When applying cross attention between decoder and encoder,
            the batch padding mask for input is in (#batch, 1, T) shape.
            2.When applying self attention of encoder,
            the mask is in (#batch, T, T)  shape.
            3.When applying self attention of decoder,
            the mask is in (#batch, L, L)  shape.
            4.If the different position in decoder see different block
            of the encoder, such as Mocha, the passed in mask could be
            in (#batch, L, T) shape. But there is no such case in current
            Wenet.
        cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
            where `cache_t == chunk_size * num_decoding_left_chunks`
            and `head * d_k == size`


    Returns:
        torch.Tensor: Output tensor (#batch, time1, d_model).
        torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
            where `cache_t == chunk_size * num_decoding_left_chunks`
            and `head * d_k == size`

    """
    q, k, v = self.forward_qkv(query, key, value)

    if cache.size(2) > 0:  # cache_t > 0
        key_cache, value_cache = torch.split(
          cache, cache.size(-1) // 2, dim=-1)
        k = torch.cat([key_cache, k], dim=2)
        v = torch.cat([value_cache, v], dim=2)
    # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
    #   non-trivial to calculate `next_cache_start` here.
    new_cache = torch.cat((k, v), dim=-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
  """Multi-Head Attention layer with relative position encoding.
  Paper: https://arxiv.org/abs/1901.02860
  Args:
    n_head (int): The number of heads.
    n_feat (int): The number of features.
    dropout_rate (float): Dropout rate.
  """
  def __init__(self, n_head, n_feat, dropout_rate):
    """Construct an RelPositionMultiHeadedAttention object."""
    super().__init__(n_head, n_feat, dropout_rate)
    # linear transformation for positional encoding
    self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
    # these two learnable bias are used in matrix c and matrix d
    # as described in https://arxiv.org/abs/1901.02860 Section 3.3
    self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
    self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
    torch.nn.init.xavier_uniform_(self.pos_bias_u)
    torch.nn.init.xavier_uniform_(self.pos_bias_v)

  def rel_shift(self, x, zero_triu: bool = False):
    """Compute relative positinal encoding.
    Args:
      x (torch.Tensor): Input tensor (batch, time, size).
      zero_triu (bool): If true, return the lower triangular part of
        the matrix.
    Returns:
      torch.Tensor: Output tensor.
    """

    zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                 device=x.device,
                 dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)

    x_padded = x_padded.view(x.size()[0],
                 x.size()[1],
                 x.size(3) + 1, x.size(2))
    x = x_padded[:, :, 1:].view_as(x)

    if zero_triu:
      ones = torch.ones((x.size(2), x.size(3)))
      x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

    return x

  def forward(self, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor,
              mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
              pos_emb: torch.Tensor = torch.empty(0),
              cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
             ):
    """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
    Args:
      query (torch.Tensor): Query tensor (#batch, time1, size).
      key (torch.Tensor): Key tensor (#batch, time2, size).
      value (torch.Tensor): Value tensor (#batch, time2, size).
      mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
        (#batch, time1, time2), (0, 0, 0) means fake mask.
      pos_emb (torch.Tensor): Positional embedding tensor
        (#batch, time2, size).
      cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
        where `cache_t == chunk_size * num_decoding_left_chunks`
        and `head * d_k == size`
    Returns:
      torch.Tensor: Output tensor (#batch, time1, d_model).
      torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
        where `cache_t == chunk_size * num_decoding_left_chunks`
        and `head * d_k == size`
    """
    q, k, v = self.forward_qkv(query, key, value)
    q = q.transpose(1, 2)  # (batch, time1, head, d_k)

    if cache.size(2) > 0:  # cache_t > 0
      key_cache, value_cache = torch.split(
        cache, cache.size(-1) // 2, dim=-1)
      k = torch.cat([key_cache, k], dim=2)
      v = torch.cat([value_cache, v], dim=2)
    # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
    #   non-trivial to calculate `next_cache_start` here.
    new_cache = torch.cat((k, v), dim=-1)

    n_batch_pos = pos_emb.size(0)
    p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
    p = p.transpose(1, 2)  # (batch, head, time1, d_k)

    # (batch, head, time1, d_k)
    q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
    # (batch, head, time1, d_k)
    q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

    # compute attention score
    # first compute matrix a and matrix c
    # as described in https://arxiv.org/abs/1901.02860 Section 3.3
    # (batch, head, time1, time2)
    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

    # compute matrix b and matrix d
    # (batch, head, time1, time2)
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    # Remove rel_shift since it is useless in speech recognition,
    # and it requires special attention for streaming.
    # matrix_bd = self.rel_shift(matrix_bd)

    scores = (matrix_ac + matrix_bd) / math.sqrt(
      self.d_k)  # (batch, head, time1, time2)

    return self.forward_attention(v, scores, mask), new_cache



class PositionalEncoding(torch.nn.Module):
  """Positional encoding.

  :param int d_model: embedding dim
  :param float dropout_rate: dropout rate
  :param int max_len: maximum input length

  PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
  PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
  """
  def __init__(self,
               d_model: int,
               dropout_rate: float,
               max_len: int = 5000,
               reverse: bool = False):
    """Construct an PositionalEncoding object."""
    super().__init__()
    self.d_model = d_model
    self.xscale = math.sqrt(self.d_model)
    self.dropout = torch.nn.Dropout(p=dropout_rate)
    self.max_len = max_len

    self.pe = torch.zeros(self.max_len, self.d_model)
    position = torch.arange(0, self.max_len,
                dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, self.d_model, 2, dtype=torch.float32) *
      -(math.log(10000.0) / self.d_model))
    self.pe[:, 0::2] = torch.sin(position * div_term)
    self.pe[:, 1::2] = torch.cos(position * div_term)
    self.pe = self.pe.unsqueeze(0)

  def forward(self, x: torch.Tensor, offset: int = 0):
    """Add positional encoding.

    Args:
      x (torch.Tensor): Input. Its shape is (batch, time, ...)
      offset (int): position offset

    Returns:
      torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
      torch.Tensor: for compatibility to RelPositionalEncoding
    """
    assert offset + x.size(1) < self.max_len
    self.pe = self.pe.to(x.device)
    pos_emb = self.pe[:, offset:offset + x.size(1)]
    x = x * self.xscale + pos_emb
    return self.dropout(x), self.dropout(pos_emb)

  def position_encoding(self, offset: int, size: int):
    """ For getting encoding in a streaming fashion

    Attention!!!!!
    we apply dropout only once at the whole utterance level in a none
    streaming way, but will call this function several times with
    increasing input size in a streaming scenario, so the dropout will
    be applied several times.

    Args:
      offset (int): start offset
      size (int): requried size of position encoding

    Returns:
      torch.Tensor: Corresponding encoding
    """
    assert offset + size < self.max_len
    return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
  """Relative positional encoding module.
  See : Appendix B in https://arxiv.org/abs/1901.02860
  Args:
      d_model (int): Embedding dimension.
      dropout_rate (float): Dropout rate.
      max_len (int): Maximum input length.
  """
  def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
    """Initialize class."""
    super().__init__(d_model, dropout_rate, max_len, reverse=True)

  def forward(self, x: torch.Tensor, offset: int = 0):
    """Compute positional encoding.
    Args:
        x (torch.Tensor): Input tensor (batch, time, `*`).
    Returns:
        torch.Tensor: Encoded tensor (batch, time, `*`).
        torch.Tensor: Positional embedding tensor (1, time, `*`).
    """
    # print("position", x.size())
    assert offset + x.size(1) < self.max_len
    self.pe = self.pe.to(x.device)
    x = x * self.xscale
    pos_emb = self.pe[:, offset:offset + x.size(1)]
    return self.dropout(x), self.dropout(pos_emb)
