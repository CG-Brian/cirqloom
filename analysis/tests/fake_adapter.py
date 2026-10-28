"""A weight-free adapter whose heads react to input content (not a test module).

    L0H1  previous-token: row q attends to q - 1
    L1H2  induction: row q attends to the position after the latest earlier occurrence of
          token[q]; with no earlier occurrence it attends to BOS (position 0)
    other heads: uniform over k <= q

Distinct, off-origin heads catch layer/head mix-ups in flattening.
"""

import numpy as np

from cirqloom_analysis.models.base import AdapterInfo, BaseModelAdapter

BOS = 9_999
INFO = AdapterInfo(
    model_id="toy",
    hf_id="example/toy",
    revision="rev0",
    tokenizer_id="example/toy",
    n_layers=2,
    n_heads=3,
    n_ctx=128,
    vocab_size=10_000,
    bos_token_id=BOS,
)
PREVIOUS_TOKEN_HEAD = (0, 1)
INDUCTION_HEAD = (1, 2)


class ContentAwareAdapter(BaseModelAdapter):
    @property
    def info(self) -> AdapterInfo:
        return INFO

    def tokenize(self, text: str) -> np.ndarray:
        return np.array([BOS, *[int(t) for t in text.split()]], dtype=np.int64)

    def token_strings(self, token_ids: np.ndarray) -> list[str]:
        return ["<bos>" if i == BOS else f" t{int(i)}" for i in token_ids]

    def _run_attention(self, token_ids: np.ndarray) -> np.ndarray:
        batch, seq = token_ids.shape
        lower = np.tril(np.ones((seq, seq)))
        uniform = lower / lower.sum(axis=-1, keepdims=True)
        out = np.broadcast_to(uniform, (batch, INFO.n_layers, INFO.n_heads, seq, seq)).copy()
        for b in range(batch):
            prev = np.zeros((seq, seq))
            prev[0, 0] = 1
            prev[np.arange(1, seq), np.arange(seq - 1)] = 1
            out[b, PREVIOUS_TOKEN_HEAD[0], PREVIOUS_TOKEN_HEAD[1]] = prev

            ind = np.zeros((seq, seq))
            for q in range(seq):
                earlier = [j for j in range(q) if token_ids[b, j] == token_ids[b, q]]
                ind[q, earlier[-1] + 1 if earlier else 0] = 1
            out[b, INDUCTION_HEAD[0], INDUCTION_HEAD[1]] = ind
        return out.astype(np.float32)
