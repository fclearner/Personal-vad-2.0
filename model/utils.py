import torch


class Swish(torch.nn.Module):
  """Construct an Swish object."""
  def forward(self, x: torch.Tensor):
    """Return Swish activation function."""
    return x * torch.sigmoid(x)


def get_activation(act):
  """Return activation function."""
  # Lazy load to avoid unused import

  activation_funcs = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU
  }

  return activation_funcs[act]()


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0):
  """Make mask tensor containing indices of padded part.

  See description of make_non_pad_mask.

  Args:
    lengths (torch.Tensor): Batch of lengths (B,).
  Returns:
    torch.Tensor: Mask tensor containing indices of padded part.

  Examples:
    >>> lengths = [5, 3, 2]
    >>> make_pad_mask(lengths)
    masks = [[0, 0, 0, 0 ,0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1]]
  """
  # lengths = torch.randint(5, (8,512))
  batch_size = lengths.size(0)
  max_len = max_len if max_len > 0 else lengths.max().item()
  seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
  seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
  seq_length_expand = lengths.unsqueeze(-1)
  mask = seq_range_expand >= seq_length_expand
  return mask
