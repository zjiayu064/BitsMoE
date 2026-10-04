"""Cache lookup restricted to context positions that do not need logits."""


def get_computed_blocks(manager, request):
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    params = request.sampling_params
    start = (params.extra_args or {}).get("bitsmoe_score_from") if params else None
    if start is None or params.prompt_logprobs is None or not manager.enable_caching:
        return manager._bitsmoe_original_get_computed_blocks(request)
    if type(start) is not int or not 1 <= start <= len(request.prompt_token_ids):
        raise ValueError("Invalid bitsmoe_score_from")
    # Recompute the hidden state that predicts the first scored token.
    blocks, count = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, min(start - 1, request.num_tokens - 1),
    )
    if manager.log_stats:
        stats = manager.prefix_cache_stats
        stats.requests += 1
        stats.queries += request.num_tokens
        stats.hits += count
    return KVCacheBlocks(blocks), count
