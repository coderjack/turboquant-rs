---
name: TurboQuant Paper Research
description: Algorithm details from Google's TurboQuant paper and blog — PolarQuant is a step within TurboQuant, QJL corrects the residual
type: reference
---

Paper: Zandieh, Daliri, Hadian, Mirrokni (2025). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874.

Blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

## Key Findings (confirmed 2026-04-04)

1. **PolarQuant IS a step within TurboQuant** — it is the main compression stage, NOT a separate competing algorithm. The blog explicitly describes it as the first stage.

2. **Random rotation is separate from PolarQuant** — rotation is preprocessing that simplifies geometry so PolarQuant can use a fixed grid. In our code, rotation is its own module.

3. **QJL is applied to the RESIDUAL** after PolarQuant — NOT to original vectors. It uses 1 bit per dimension. The correction gives an unbiased inner product estimator.

4. **Full pipeline:** Rotate → PolarQuant (majority of bits) → QJL on residual (1 bit)

5. **Unbiased estimator:** `<q,x> ≈ ‖x‖ · (<q_rot, ŷ> + ‖r‖ · √(π/(2d)) · Σ sign_i · (S·q_rot)_i)`

## Paper also has TurboQuant_mse (Algorithm 1)

Uses Lloyd-Max scalar quantization instead of polar coordinates. We chose the PolarQuant-based variant as described in the blog.

## PolarQuant (separate paper)

Han et al., 2025, arXiv:2502.02617. Describes the polar coordinate quantization technique that TurboQuant incorporates.
