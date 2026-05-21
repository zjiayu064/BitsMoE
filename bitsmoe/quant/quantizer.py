import torch
from torch import Tensor
from typing import Dict, Tuple

from bitsmoe.quant.bitpack import BitPack


def svd_decompose(weight, to_cpu=False):
    dtype = weight.dtype
    w = weight.float()
    u, s, v = torch.linalg.svd(w, full_matrices=False)
    u, s, v = u.to(dtype), s.to(dtype), v.to(dtype)
    if to_cpu:
        return u.cpu(), s.cpu(), v.cpu()
    return u, s, v

def pack_quantized(W_q: Tensor, bits: int):
    """
    Bit-pack an int8 quantized tensor using BitPack.
    """
    return BitPack.pack(W_q, bits)

def unpack_quantized(W_q: Tensor, bits: int, original_size: int):
    """
    Unpack a bit-packed quantized integer tensor back into its original dense
    integer representation.
    """
    return BitPack.unpack(W_q, bits, original_size)

def quantize_symmetric(x, bits=8, dim=0, eps=1e-8, iters=50):
    """
    Symmetric signed quantization with per-channel scaling.

    Args:
        x (torch.Tensor): Input tensor of shape [M, N].
        bits (int): Quantization bit-width. bits=1 triggers binary quantization.
        dim (int): Channel dimension used to compute scale.
                   dim=0 → per-column quantization (typical for Linear weights)
                   dim=1 → per-row quantization.
        eps (float): Numerical stability epsilon.

    Returns:
        x_quant (torch.IntTensor): Quantized tensor with the same shape as x.
        scale (torch.Tensor): Per-channel scale factors.
    """
    if bits <= 0:
        raise ValueError(f"bits must be in (0, 8], got {bits}")
    if x.numel() == 0:
        return x.to(torch.int8), torch.tensor(1.0, device=x.device)

    # Binary quantization (1-bit)
    if bits == 1:
        scale = x.abs().mean(dim=dim, keepdim=True).clamp(min=eps)
        x_sign = torch.sign(x)
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_quant = x_sign.to(torch.int8)
        return x_quant, scale.squeeze(dim)

    # Multi-bit symmetric quantization
    elif bits in [2, 3, 4]:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1

        scale = x.abs().mean(dim=dim, keepdim=True).clamp(min=eps)

        for _ in range(iters):
            q = (x / scale).round().clamp(qmin, qmax)
            num = (x * q).mean(dim=dim, keepdim=True)
            den = (q * q).mean(dim=dim, keepdim=True)
            scale = (num / (den + eps)).clamp(min=eps)

        x_quant = q.to(torch.int8)
        scale = scale.squeeze(dim)
    else:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1

        abs_max = x.abs().amax(dim=dim, keepdim=True)
        scale = abs_max / qmax
        scale = scale.clamp(min=eps)

        x_quant = (x / scale).round().clamp(qmin, qmax).to(torch.int8)
        scale = scale.squeeze(dim)
    return x_quant, scale


def quantize_symmetric_block(x, bits=8, dim=0, eps=1e-8):
    """
    Symmetric signed quantization for block-ILP path.

    bits=1 keeps binary quantization.
    bits>1 always uses absmax scaling on the target channel dimension.
    """
    if bits <= 0:
        raise ValueError(f"bits must be in (0, 8], got {bits}")
    if x.numel() == 0:
        return x.to(torch.int8), torch.tensor(1.0, device=x.device)

    if bits == 1:
        scale = x.abs().mean(dim=dim, keepdim=True).clamp(min=eps)
        x_sign = torch.sign(x)
        x_sign = torch.where(x_sign == 0, torch.ones_like(x_sign), x_sign)
        x_quant = x_sign.to(torch.int8)
        return x_quant, scale.squeeze(dim)

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1

    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max / qmax).clamp(min=eps)

    x_quant = (x / scale).round().clamp(qmin, qmax).to(torch.int8)
    return x_quant, scale.squeeze(dim)


def dequantize_symmetric(x_quant, scale, dim=0):
    """
    Symmetric dequantization.

    Args:
        x_quant (torch.IntTensor): Quantized tensor.
        scale (torch.Tensor): Per-channel scale factor.
        dim (int): Same channel dimension used during quantization.

    Returns:
        torch.Tensor: Dequantized tensor in float16.
    """
    return (x_quant.to(torch.float32) * scale.to(torch.float32).unsqueeze(dim)).to(torch.float16)


@torch.no_grad()
def estimate_u_low_bit_error_factors(
    U: Tensor,
    bits: Tuple[int, ...] = (2, 3, 4),
    group_size: int = 128,
) -> Dict[int, Tensor]:
    """
    Estimate per-column quantization error factors for U using the exact same
    quantization flow as fake_quantize (grouped along rows, per-column channels).

    For each bit in ``bits``, returns:
        err_b[k] = ||U[:, k] - U_q[:, k]||_2^2
    """
    if U.dim() != 2:
        raise ValueError(f"U must be 2-D, got shape={tuple(U.shape)}")

    m, r = U.shape
    if m % group_size != 0:
        raise ValueError(
            f"U.shape[0]={m} is not divisible by group_size={group_size}"
        )

    U_f32 = U.float()
    U_view = U_f32.reshape(m // group_size, group_size, r)
    out: Dict[int, Tensor] = {}

    for bit in bits:
        U_int, U_scale = quantize_symmetric(U_view, bits=bit, dim=1)
        U_deq = dequantize_symmetric(U_int, U_scale, dim=1).reshape(m, r).float()
        out[int(bit)] = ((U_f32 - U_deq) ** 2).sum(dim=0).double()  # [rank]

    return out


@torch.no_grad()
def estimate_vh_low_bit_error_factors(
    Vh: Tensor,
    bits: Tuple[int, ...] = (2, 3, 4),
    group_size: int = 128,
) -> Dict[int, Tensor]:
    """
    Estimate per-row quantization error factors for Vh with grouped quantization
    along its feature dimension.

    For each bit in ``bits``, returns:
        err_b[k] = ||Vh[k, :] - Vh_q[k, :]||_2^2
    """
    if Vh.dim() != 2:
        raise ValueError(f"Vh must be 2-D, got shape={tuple(Vh.shape)}")

    r, n = Vh.shape
    if n % group_size != 0:
        raise ValueError(
            f"Vh.shape[1]={n} is not divisible by group_size={group_size}"
        )

    Vh_f32 = Vh.float()
    Vh_view = Vh_f32.reshape(r, n // group_size, group_size)
    out: Dict[int, Tensor] = {}

    for bit in bits:
        Vh_int, Vh_scale = quantize_symmetric(Vh_view, bits=bit, dim=2)
        Vh_deq = dequantize_symmetric(Vh_int, Vh_scale, dim=2).reshape(r, n).float()
        out[int(bit)] = ((Vh_f32 - Vh_deq) ** 2).sum(dim=1).double()  # [rank]

    return out
