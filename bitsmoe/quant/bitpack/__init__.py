import torch
from torch import Tensor
from typing import Dict, Tuple

from bitsmoe.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from bitsmoe.quant import _bitpack_cuda as bitpack_cuda
    _HAS_CUDA_BITPACK = True
    logger.info("CUDA bitpack extension loaded successfully.")
except Exception as e:
    bitpack_cuda = None
    _HAS_CUDA_BITPACK = False
    raise RuntimeError("CUDA bitpack extension is required for BitPack.") from e


class BitPack:
    """CUDA-only bit packing with uint32 bit-plane layout."""

    BIT_OFFSETS: Dict[int, int] = {1: 1, 2: 2, 3: 4, 4: 8, 6: 32, 8: 128}

    @staticmethod
    def _check_bits(bits: int) -> None:
        if bits not in BitPack.BIT_OFFSETS:
            raise ValueError(f"Unsupported bit width: {bits}")

    @staticmethod
    def _check_pack_input(W_q: Tensor) -> Tensor:
        if not _HAS_CUDA_BITPACK:
            raise RuntimeError("CUDA bitpack extension is not available")
        if W_q.dim() != 2:
            raise ValueError(f"BitPack.pack expects 2D tensor, got shape={tuple(W_q.shape)}")
        if W_q.dtype != torch.int8:
            raise TypeError(f"BitPack.pack only supports int8 input, got {W_q.dtype}")
        if not W_q.is_cuda:
            raise ValueError("BitPack.pack requires CUDA tensor input")
        return W_q.contiguous() if not W_q.is_contiguous() else W_q

    @staticmethod
    def _check_unpack_input(W_q: Tensor) -> Tensor:
        if not _HAS_CUDA_BITPACK:
            raise RuntimeError("CUDA bitpack extension is not available")
        if W_q.dim() != 2:
            raise ValueError(f"BitPack.unpack expects 2D tensor, got shape={tuple(W_q.shape)}")
        if W_q.dtype != torch.uint32:
            raise TypeError(f"BitPack.unpack expects uint32 packed tensor, got {W_q.dtype}")
        if not W_q.is_cuda:
            raise ValueError("BitPack.unpack requires CUDA tensor input")
        return W_q.contiguous() if not W_q.is_contiguous() else W_q

    @staticmethod
    def _int8_to_uint8(W_q: Tensor, bits: int) -> Tensor:
        return bitpack_cuda.int8_to_uint8(W_q, BitPack.BIT_OFFSETS[bits])

    @staticmethod
    def _uint8_to_int8(W_q: Tensor, bits: int) -> Tensor:
        return bitpack_cuda.uint8_to_int8(W_q, BitPack.BIT_OFFSETS[bits])

    @staticmethod
    def pack(W_q: Tensor, bits: int) -> Tuple[Tensor, int]:
        """Pack int8 tensor into uint32 bit-plane layout."""
        BitPack._check_bits(bits)
        W_q = BitPack._check_pack_input(W_q)

        # Fast path: fused signed-int8 pack kernels (avoid extra int8->uint8 pass).
        if hasattr(bitpack_cuda, "pack_int8_1bit"):
            dispatch_i8 = {
                1: bitpack_cuda.pack_int8_1bit,
                2: bitpack_cuda.pack_int8_2bit,
                3: bitpack_cuda.pack_int8_3bit,
                4: bitpack_cuda.pack_int8_4bit,
                6: bitpack_cuda.pack_int8_6bit,
                8: bitpack_cuda.pack_int8_8bit,
            }
            packed = dispatch_i8[bits](W_q)
        else:
            # Backward-compatible path for older compiled extensions.
            unsigned = BitPack._int8_to_uint8(W_q, bits)
            dispatch_u8 = {
                1: bitpack_cuda.pack_1bit,
                2: bitpack_cuda.pack_2bit,
                3: bitpack_cuda.pack_3bit,
                4: bitpack_cuda.pack_4bit,
                6: bitpack_cuda.pack_6bit,
                8: bitpack_cuda.pack_8bit,
            }
            packed = dispatch_u8[bits](unsigned)

        return packed, W_q.shape[1]

    @staticmethod
    def unpack(W_q: Tensor, bits: int, original_size: int) -> Tensor:
        """Unpack uint32 bit-plane tensor back to int8."""
        BitPack._check_bits(bits)
        W_q = BitPack._check_unpack_input(W_q)

        if original_size < 0:
            raise ValueError(f"original_size must be non-negative, got {original_size}")

        # Fast path: fused unpack-to-int8 kernels.
        if hasattr(bitpack_cuda, "unpack_int8_1bit"):
            dispatch_i8 = {
                1: bitpack_cuda.unpack_int8_1bit,
                2: bitpack_cuda.unpack_int8_2bit,
                3: bitpack_cuda.unpack_int8_3bit,
                4: bitpack_cuda.unpack_int8_4bit,
                6: bitpack_cuda.unpack_int8_6bit,
                8: bitpack_cuda.unpack_int8_8bit,
            }
            out = dispatch_i8[bits](W_q)
            if original_size > out.shape[1]:
                raise ValueError(
                    f"original_size={original_size} exceeds unpacked width={out.shape[1]}"
                )
            return out[:, :original_size].contiguous()

        # Backward-compatible path for older compiled extensions.
        dispatch_u8 = {
            1: bitpack_cuda.unpack_1bit,
            2: bitpack_cuda.unpack_2bit,
            3: bitpack_cuda.unpack_3bit,
            4: bitpack_cuda.unpack_4bit,
            6: bitpack_cuda.unpack_6bit,
            8: bitpack_cuda.unpack_8bit,
        }
        unsigned = dispatch_u8[bits](W_q)
        if original_size > unsigned.shape[1]:
            raise ValueError(
                f"original_size={original_size} exceeds unpacked width={unsigned.shape[1]}"
            )
        unsigned = unsigned[:, :original_size].contiguous()
        return BitPack._uint8_to_int8(unsigned, bits)


__all__ = ["BitPack"]
