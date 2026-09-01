# FlashInfer probability-CDF sampling tail bias in MCore

## Summary

Megatron inference rollouts can emit extremely low-probability tokens much more
often than their reported logprobs imply. The logprobs themselves are correct;
the bias is in token selection.

NeMo-RL previously forced MCore's FlashInfer sampling backend and passed the
public no-filter value `top_p=1.0`. MCore treats only `top_p=0.0` as disabled,
so the default request entered `top_p_sampling_from_probs` instead of an
unfiltered logits sampler.

FlashInfer samples that float32 probability tensor with a cumulative
distribution. If reduction rounding leaves the accumulated mass below a random
draw close to one, its kernel falls back to the last vocabulary ID with positive
probability. That ID can have a real logprob of -20 to -40. The selected token's
reported logprob therefore agrees between inference and training even though
the token was chosen by the fallback rather than at its model probability.

vLLM does not show the same tail because it maps `top_p=1.0` to no filter and
uses its native exponential/Gumbel sampling path when neither top-k nor top-p
is active.

## GPU confirmation

Run the standalone reproducer in an MCore environment with a CUDA GPU:

```bash
uv run --extra mcore \
  tools/model_diagnostics/7.flashinfer_sampling_tail_bias.py
```

On a GB200 with `flashinfer-python==0.6.8.post1`, 5 million draws produced:

| Sampler | Samples of the final ID | Observed rate |
| --- | ---: | ---: |
| `top_p_sampling_from_probs(p=1)` | 14 | 2.8e-6 |
| `sampling_from_probs` | 15 | 3.0e-6 |
| `sampling_from_logits` | 0 | 0 |
| `torch.multinomial` | 0 | 0 |

The constructed final token had logprob -40 and a correct expected count of
`2.12e-11`. The probability tensor's float32 mass shortfall was `3.10e-6`,
which predicts about 15.5 fallback selections and matches the observed result.

## Fix

[Megatron generation](../../nemo_rl/models/generation/megatron/megatron_worker.py)
now uses MCore's Torch sampling backend for correctness. It also normalizes the
public no-filter spelling `top_p=1.0` to MCore's `top_p=0.0` sentinel, avoiding a
no-op nucleus-filter path.

The durable high-performance upstream fix is to use FlashInfer's
`sampling_from_logits` for the unfiltered case, or otherwise eliminate the
last-valid-ID fallback bias. Until that lands in MCore, opting back into the
FlashInfer probability samplers risks corrupting long RL rollouts.

## Checking existing rollout data

As a fast confirmation, inspect token IDs for samples with very low reported
logprobs. This bug predicts that they concentrate on the last positive
vocabulary ID (usually the highest vocabulary ID), rather than being spread
across low-probability tokens.

For a full calibration test, compute the total probability mass in each logprob
bin at every sampling step and sum that mass across steps. Compare those
expected bin counts with the observed sampled-token counts. Using `exp(logprob)`
for one emitted token as the per-draw expectation omits the other vocabulary
tokens in the same bin and overstates the statistical significance.
