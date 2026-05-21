from typing import Dict, Optional

import torch
from bitsmoe.models.base_mlp import BitsMoE_BaseMoeMLP


class BitsMoE_DeepSeekMLP(BitsMoE_BaseMoeMLP):
    def __init__(
        self,
        config,
        intermediate_size: Optional[int] = None,
        packed_state: Optional[Dict[str, torch.Tensor]] = None,
        packed_prefix: Optional[str] = None,
        packed_device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            config=config,
            intermediate_size=intermediate_size,
            packed_state=packed_state,
            packed_prefix=packed_prefix,
            packed_device=packed_device,
        )
