# Qwen3 Model Notes

This directory contains the high-level Qwen3 model wrapper. The implementation supports Hugging Face checkpoints with
`model_type = "qwen3"`, tied word embeddings, dense decoder layers, grouped-query causal attention, RoPE, q/k RMSNorm,
and the Qwen gated MLP.

## Symbols

| Symbol | Meaning |
| --- | --- |
| `seq` | Number of token positions processed by the current call. During cached generation this is usually `1`. |
| `cached_seq` | Number of token positions already stored in the K/V cache before the current call. |
| `key_seq` | Number of source positions visible to attention, normally `cached_seq + seq`. |
| `hidden` | Model hidden size from `config.hidden_size`. |
| `vocab` | Vocabulary size from `config.vocab_size`. |
| `q_heads` | Query attention heads from `config.num_attention_heads`. |
| `kv_heads` | Key/value heads from `config.num_key_value_heads`. |
| `head_dim` | Per-head width from `config.head_dim`. |
| `intermediate` | MLP expansion width from `config.intermediate_size`. |

Qwen uses grouped-query attention when `q_heads > kv_heads`. In that case several query heads share the same K/V head
before K/V are repeated to match `q_heads` for the attention matmuls.

## Full Model Flow

The public model input is token ids:

```text
token_ids: [seq]
```

The model then runs:

```text
embedding lookup                    [seq] -> [seq, hidden]
decoder block 0                     [seq, hidden] -> [seq, hidden]
...
decoder block N - 1                 [seq, hidden] -> [seq, hidden]
final RMSNorm                       [seq, hidden] -> [seq, hidden]
tied embedding output projection    [seq, hidden] x [hidden, vocab] -> [seq, vocab]
```

`Qwen3Model::forward()` creates a temporary cache and processes the whole sequence. `Qwen3Session` owns the processed
token ids plus a persistent cache, so callers can pass the complete growing token sequence while the model forwards
only the uncached suffix.

## Decoder Block

Each decoder block is pre-norm and has two residual paths:

```text
hidden_states                       [seq, hidden]
input RMSNorm                       [seq, hidden]
self-attention                      [seq, hidden]
attention residual add              [seq, hidden]
post-attention RMSNorm              [seq, hidden]
MLP                                 [seq, hidden]
MLP residual add                    [seq, hidden]
```

The block output keeps the same shape and dtype as the block input.

## Attention

The attention input is always rank-2:

```text
hidden_states: [seq, hidden]
```

The bias-free projections produce packed head dimensions:

```text
q_proj: [seq, q_heads * head_dim]
k_proj: [seq, kv_heads * head_dim]
v_proj: [seq, kv_heads * head_dim]
```

Those packed tensors are reshaped into explicit heads:

```text
query_heads: [seq, q_heads, head_dim]
key_heads:   [seq, kv_heads, head_dim]
value_heads: [seq, kv_heads, head_dim]
```

Qwen applies RMSNorm to each query/key head and RoPE to the normalized query/key tensors:

```text
normalized_query: [seq, q_heads, head_dim]
normalized_key:   [seq, kv_heads, head_dim]
rotated_query:    [seq, q_heads, head_dim]
rotated_key:      [seq, kv_heads, head_dim]
```

The current call appends only new keys and values to the per-layer cache:

```text
cached_key:   [key_seq, kv_heads, head_dim]
cached_value: [key_seq, kv_heads, head_dim]
```

Inside `fused_causal_attention`, tensors are permuted into head-major matmul batches:

```text
query_batch:      [q_heads, seq, head_dim]
key_by_kv_head:   [kv_heads, head_dim, key_seq]
value_by_kv_head: [kv_heads, key_seq, head_dim]
key_batch:        [q_heads, head_dim, key_seq]
value_batch:      [q_heads, key_seq, head_dim]
```

After repeating K/V heads for grouped-query attention, scores and probabilities are:

```text
attention_scores: [q_heads, seq, key_seq]
probabilities:    [q_heads, seq, key_seq]
```

The causal mask is applied in absolute token positions, so a cached one-token decode at position `p` may attend to keys
`0..p` and future key positions stay zero after softmax. The value matmul produces:

```text
batch_result:     [q_heads, seq, head_dim]
attention_output: [seq, q_heads, head_dim]
packed_attention: [seq, q_heads * head_dim]
attention_result: [seq, hidden]
```

`packed_attention` is the attention output with `q_heads` and `head_dim` flattened again. That gives the output
projection the standard linear weight shape.

## MLP

The Qwen MLP is the gated `SiLU(gate) * up` form:

```text
hidden_states:     [seq, hidden]
gate_projection:   [seq, intermediate]
up_projection:     [seq, intermediate]
activated_gate:    [seq, intermediate]
gated_projection:  [seq, intermediate]
down_projection:   [seq, hidden]
```

All three MLP projections are bias-free in the supported checkpoints.

## Cache

Each decoder layer owns one attention cache:

```text
key_cache:   [capacity, kv_heads, head_dim]
value_cache: [capacity, kv_heads, head_dim]
```

The filled prefix is exposed as:

```text
key_view:   [sequence_length, kv_heads, head_dim]
value_view: [sequence_length, kv_heads, head_dim]
```

The cache can start empty and allocate on first append, or be preallocated through `Qwen3Model::make_cache(max_seq)`.
When more capacity is needed, storage grows and the filled prefix is preserved.
