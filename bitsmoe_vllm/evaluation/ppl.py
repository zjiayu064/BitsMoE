import math
from collections.abc import Iterator, Sequence


def strided_windows(
    token_ids: Sequence[int], max_length: int, stride: int,
) -> Iterator[tuple[list[int], int]]:
    if len(token_ids) < 2 or max_length < 2 or not 1 <= stride <= max_length:
        raise ValueError("Need at least two tokens, max_length >= 2, and 1 <= stride <= max_length")
    previous_end = 0
    for begin in range(0, len(token_ids), stride):
        end = min(begin + max_length, len(token_ids))
        target_length = end - previous_end
        window = list(token_ids[begin:end])
        yield window, max(1, len(window) - target_length)
        previous_end = end
        if end == len(token_ids):
            break


def compute_ppl_vllm_windows(engine, windows, batch_size=1, selective=False) -> float:
    from vllm import SamplingParams

    if batch_size < 1:
        raise ValueError("PPL batch size must be positive")
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    nll, count = 0.0, 0
    windows = list(windows)
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        if selective:
            from ..runtime.scoring import score_requests

            results = score_requests(engine, [w for w, _ in batch], [s for _, s in batch])
        else:
            results = engine.generate([{"prompt_token_ids": w} for w, _ in batch],
                                      params, use_tqdm=False)
        for (window, score_from), result in zip(batch, results):
            if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(window):
                raise RuntimeError("vLLM did not return prompt likelihoods; disable prefix caching")
            for index in range(score_from, len(window)):
                row = result.prompt_logprobs[index]
                if row is None or window[index] not in row:
                    raise RuntimeError(f"Missing prompt likelihood at token {index}")
                nll -= row[window[index]].logprob
                count += 1
    if not count:
        raise ValueError("No PPL targets to score")
    return math.exp(nll / count)


def compute_ppl_vllm_strided(engine, token_ids: Sequence[int], max_length: int, stride: int,
                           batch_size=1, selective=False) -> float:
    return compute_ppl_vllm_windows(
        engine, strided_windows(token_ids, max_length, stride), batch_size, selective,
    )
