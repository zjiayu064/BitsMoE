"""BitsMoE package init."""

# Ensure AutoModelForCausalLM.from_pretrained can auto-detect BitsMoE checkpoints.
from bitsmoe.models.auto.bitsmoe_auto_patch import ensure_auto_patch_installed

ensure_auto_patch_installed()
