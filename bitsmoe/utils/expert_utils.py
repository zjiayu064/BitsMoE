import torch


PROJ_NAME_MAP = {
    # canonical
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "down_proj": "down_proj",

    # Mixtral
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}

def infer_expert_mtypes(expert):
    preferred = ["gate_proj", "up_proj", "down_proj"]
    fallback = ["w1", "w2", "w3"]

    if all(hasattr(expert, n) for n in preferred):
        return preferred

    if all(hasattr(expert, n) for n in fallback):
        return fallback

    # Collect Linear layers for debug
    available = [
        name for name, m in expert.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]

    raise RuntimeError(
        "Failed to infer expert FFN projection layers. "
        "Expected either MoE-style projections "
        "['gate_proj', 'up_proj', 'down_proj'] "
        "or Mixtral-style projections ['w1', 'w2', 'w3'], "
        f"but found Linear layers: {available}."
    )

def map_mtype(mtype: str) -> str:
    """
    Map expert projection name to canonical MoE naming.

    Canonical names:
        - gate_proj
        - up_proj
        - down_proj
    """
    if mtype not in PROJ_NAME_MAP:
        raise RuntimeError(
            f"Unknown expert projection name '{mtype}'. "
            "Expected one of: gate_proj, up_proj, down_proj, w1, w2, w3."
        )

    return PROJ_NAME_MAP[mtype]


def extract_expert_weights(expert):
    weights = {}

    for attr_name, canonical_name in PROJ_NAME_MAP.items():
        if hasattr(expert, attr_name):
            proj = getattr(expert, attr_name)
            if hasattr(proj, "weight"):
                weights[canonical_name] = proj.weight.detach()
    
    # sanity check
    if len(weights) != 3:
        raise RuntimeError(
            f"Expert projection mismatch, got {list(weights.keys())} "
            f"from {expert.__class__.__name__}"
        )
    return weights

def get_moe_block(layer):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
        return layer.block_sparse_moe
    return None
