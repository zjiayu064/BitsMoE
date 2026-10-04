"""Continuation scoring without computing context vocabulary probabilities."""


def score_requests(engine, token_ids, score_from, *, reuse_prefixes=False):
    from vllm import SamplingParams

    if len(token_ids) != len(score_from):
        raise ValueError("Each request needs a scoring offset")
    params = []
    for tokens, start in zip(token_ids, score_from):
        if type(start) is not int or not 1 <= start <= len(tokens):
            raise ValueError("Scoring offsets must be between 1 and the prompt length")
        params.append(SamplingParams(
            temperature=0, max_tokens=1, prompt_logprobs=1, detokenize=False,
            extra_args={"bitsmoe_score_from": start},
        ))
    if reuse_prefixes:
        from collections import Counter

        config = engine.llm_engine.vllm_config
        if not config.cache_config.enable_prefix_caching:
            raise ValueError("Prefix reuse requires enable_prefix_caching=True")
        contexts = Counter(tuple(tokens[:start]) for tokens, start in zip(token_ids, score_from))
        first, last = min(contexts, default=()), max(contexts, default=())
        common = 0
        if len(contexts) > 1:
            while common < min(len(first), len(last)) and first[common] == last[common]:
                common += 1
        if common >= 64:
            engine.generate([{"prompt_token_ids": list(first[:common])}], SamplingParams(
                temperature=0, max_tokens=1, detokenize=False), use_tqdm=False)
        shared = [list(tokens) for tokens, count in contexts.items()
                  if count > 1 and (common < 64 or len(tokens) > common)]
        if shared:
            engine.generate([{"prompt_token_ids": ids} for ids in shared], SamplingParams(
                temperature=0, max_tokens=1, detokenize=False), use_tqdm=False)
    outputs = engine.generate([{"prompt_token_ids": ids} for ids in token_ids],
                              params, use_tqdm=False)
    for output, start in zip(outputs, score_from):
        output.prompt_logprobs[:start] = [None] * start
    return outputs


def prompt_logprobs(runner, hidden_states, num_scheduled_tokens):
    from collections import deque

    import torch
    from vllm.v1.outputs import LogprobsTensors

    requested = runner.input_batch.num_prompt_logprobs
    if not any((runner.requests[key].sampling_params.extra_args or {}).get("bitsmoe_score_from")
               for key in requested):
        return runner._bitsmoe_full_prompt_logprobs(hidden_states, num_scheduled_tokens)

    pending = runner.input_batch.in_progress_prompt_logprobs_cpu
    completed, groups = {}, {}
    for key, topk in requested.items():
        request = runner.requests[key]
        tokens = request.prompt_token_ids
        if tokens is None:
            continue
        score_from = (request.sampling_params.extra_args or {}).get("bitsmoe_score_from", 1)
        if type(score_from) is not int or not 1 <= score_from <= len(tokens):
            raise ValueError("Invalid bitsmoe_score_from")
        if key not in pending:
            rows = LogprobsTensors.empty_cpu(len(tokens) - 1, topk + 1)
            rows.logprob_token_ids.zero_()
            rows.logprobs.fill_(float("nan"))
            rows.selected_token_ranks.fill_(-1)
            pending[key] = rows
        start = request.num_computed_tokens
        scheduled = num_scheduled_tokens[key]
        remaining = len(tokens) - start - 1
        if scheduled > remaining:
            completed[key] = pending[key]
        end = start + min(scheduled, max(remaining, 0))
        first = max(start, score_from - 1)
        req_index = runner.input_batch.req_id_to_index[key]
        offset = int(runner.query_start_loc.np[req_index])
        for begin in range(first, end, 1024):
            stop = min(begin + 1024, end)
            h = hidden_states[offset + begin - start:offset + stop - start]
            groups.setdefault(topk, []).append((pending[key], begin, stop, h, tokens[begin + 1:stop + 1]))

    for topk, pieces in groups.items():
        pieces = deque(pieces)
        while pieces:
            batch, count = [], 0
            while pieces and count + pieces[0][2] - pieces[0][1] <= 1024:
                piece = pieces.popleft()
                batch.append(piece)
                count += piece[2] - piece[1]
            h = torch.cat([piece[3] for piece in batch])
            targets = torch.tensor([token for piece in batch for token in piece[4]],
                                   device=runner.device, dtype=torch.int64)
            logits = runner.model.compute_logits(h)
            probs = runner.sampler.compute_logprobs(logits)
            ids, probs, ranks = runner.sampler.gather_logprobs(probs, topk, targets)
            offset = 0
            for rows, begin, stop, _, _ in batch:
                source = slice(offset, offset + stop - begin)
                rows.logprob_token_ids[begin:stop].copy_(ids[source], non_blocking=True)
                rows.logprobs[begin:stop].copy_(probs[source], non_blocking=True)
                rows.selected_token_ranks[begin:stop].copy_(ranks[source], non_blocking=True)
                offset += stop - begin

    for key in completed:
        del requested[key]
        del pending[key]
    if completed:
        runner._sync_device()
    return completed
