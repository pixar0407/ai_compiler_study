import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def torch_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    # x = s b h d
    tmp_shape = x.shape[:-1] + torch.Size((2, x.shape[-1] // 2))
    x = x.view(tmp_shape)   # x = s b h 2 d/2
    x1, x2 = x.unbind(dim=-2) # x1 = s b h d/2 , x2 = s b h d/2 ; not odd even, just first part and second part : half and half
    out = torch.cat((-x2, x1), dim=-1) # out = s b h d

    return out

def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[s, b, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )
    xx = t
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision

    cos_ = torch.cos(freqs).to(t.dtype) # s, 1, 1, d
    sin_ = torch.sin(freqs).to(t.dtype) # s, 1, 1, d    


    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t_out = (t * cos_) + (torch_rotate_half(t) * sin_)
    #t_out.retain_grad()

    out = torch.cat((t_out, t_pass), dim=-1)

    return out

##############################################
############################################## Set dimensions.
##############################################
if __name__ == '__main__':
# sequence length
# batch size
# head num
# hidden dimension
 s = 384 
 b = 3072
 h = 8 
 d = 32

##############################################
############################################## Set inputs
##############################################
 inputs = torch.rand(s,b,h,d)
 freqs  = torch.rand(s,1,1,d)

##############################################
############################################## Set outputs
##############################################
 outputs = apply_rotary_pos_emb(inputs,freqs)
 print("done")



