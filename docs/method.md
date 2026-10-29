# Method contract

ReadPrune (the paper's FRA-Q2 selector) keeps the original token vectors. It does
not perform exchange repair, NDR, token merging, or parameter training.

1. Capture the input to the final frozen CLIP visual block. For each patch,
   suppress its key connection from the CLS query and renormalize the remaining
   attention weights. The CLS self-connection and upstream states stay fixed.
2. Evaluate intact and intervened CLS outputs through the attention output
   projection, residual path, MLP, post-layer normalization, visual projection,
   and unit normalization. Define `R[i] = e_full - e_without_edge_i`.
3. Encode the raw question with the paired frozen CLIP text encoder. Long text
   is split without truncation, each chunk is unit-normalized, and the final
   unit vector is the content-length-weighted sum of chunk vectors.
4. Let `delta = R @ t`, `s = max(abs(delta) - 1e-6, 0)`, and `mu = mean(s)`.
   Set `q = 1 + s / (s + mu)`; use all ones when `mu == 0`.
5. Construct `B = U @ U.T + 1e-6 I` from row-normalized **native projector
   features**. Signed cosine values are preserved.
6. Select exactly K tokens with the kernel
   `L = diag(q**2) @ B @ diag(q**2)`. The subset objective is
   `4 sum(log(q[S])) + logdet(B[S,S])`.
7. Sort the selected indices into original spatial order and gather the
   unchanged native vectors for the decoder.

The reference selector uses CPU FP64 arithmetic, fixed ridge, deterministic
ties, and direct Cholesky adjudication near ties. It raises on numerical
inconsistency rather than silently increasing the ridge or clipping residuals.

The response is a local intervention in frozen CLIP readout space. It is not the
answer-level causal effect of deleting a token from an LVLM input.
