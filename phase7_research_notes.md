# Phase 7 Research Notes: Target Escalation & Feedback Oscillation

## Empirical Observation: The ETA Tipping Point
Based on recent training logs, a critical structural flip was observed directly correlated to the dynamic `ETA_TARGET` parameter:
*   At **Epoch 6 (`ETA = 0.40`)**, the generated images collapsed into pure black (**Blackout**).
*   At **Epoch 7-10 (`ETA = 0.50`)**, the generated images violently reversed and collapsed into pure white (**Whiteout**).

This proves that the collapse is not merely a slow, compounding error, but a violent **Optimization Oscillation**. The target update step size (`ETA = 0.50`) is far too aggressive for the highly non-linear, compounded space of the recursive DTP architecture. Instead of smoothly descending the loss landscape, the generator is over-correcting, jumping from a saturated negative `tanh` state (-1.0) completely past the data manifold and slamming into a saturated positive `tanh` state (+1.0). This is the architectural equivalent of the "Dying ReLU" problem, where massive gradient updates push the network into an irreversible, saturated mathematical space.

## Diagnosis: Target Escalation & Oscillation
While the structural safety nets implemented in Phase 6 (Top-Level Target Clamping, Label Smoothing, TTUR) successfully prevented the immediate `NaN` (Not a Number) gradient explosion, the network is now suffering from a deeper systemic issue inherent to recursive Difference Target Propagation (DTP): **Target Escalation**.

1.  **ETA_TARGET Engagement:** At Epoch 3, the `ETA_TARGET` begins ramping up, meaning the generator starts adjusting its weights based on the targets propagated backward by the localized inverse autoencoders.
2.  **Inverse Compounding Error:** The localized inverse networks (`predict_back_v2`) are non-linear approximations of the true inverse function. If an inverse mapping possesses a slight numerical bias (e.g., tending to add +0.1 to activations), this error acts as a geometric multiplier.
3.  **Unbounded Propagation:** As the target propagates downward (`T4 -> T3 -> T2 -> T1`), the slight inaccuracies compound. The requested target representations at `T1` become structurally extreme.
4.  **Feedback Oscillation:** The early generator layers update their weights to match these extreme targets, causing the output to overshoot into pure black. The discriminator harshly penalizes this, slamming the top-level target in the opposite direction. The compounding inverses amplify this reverse signal, causing the generator to slingshot into pure white.

## Proposed Mathematical Methodologies for Research

To stabilize the internal dynamics of the DTP architecture, research the following interventions:

### 1. Layer-wise Target Clamping (The Structural Fix)
*   **Theory:** Currently, only the top-level target `T_img` is restricted to `[-1.0, 1.0]`. The intermediate targets (`T4, T3, T2, T1`) are completely unbounded. If unbounded, the compounded error from the inverse mappings allows them to request massive, unreachable activation states.
*   **Implementation Strategy:** Apply `tf.clip_by_value` to *every* intermediate target immediately after it is calculated (e.g., `T3 = tf.clip_by_value(..., MIN_BOUND, MAX_BOUND)`).
*   **Research Question:** What are the mathematically appropriate bounds for the outputs of the intermediate forward blocks, given they utilize `LayerNormalization` and `LeakyReLU(0.2)`?

### 2. Extended Inverse Warmup & Decoupling (The Dynamics Fix)
*   **Theory:** The non-linear inverse autoencoders (`Conv2D -> LayerNorm -> LeakyReLU -> Conv2D`) require significant training time to accurately map the forward representations backward. A 2-epoch warmup at `ETA = 0.0` is likely insufficient. Trusting inaccurate inverses to guide the generator guarantees target corruption.
*   **Implementation Strategy:**
    *   Significantly extend the `ETA_TARGET = 0.0` warmup phase (e.g., to 10-20 epochs) to allow the inverses to converge on their local MSE reconstruction losses.
    *   Investigate asymmetric learning rates: configuring the local inverse optimizers to take larger or more frequent update steps relative to the forward generator optimizers.
*   **Research Question:** How can we mathematically verify that an inverse mapping has sufficiently converged before engaging the forward target propagation?

### 3. Target Step Capping and Gradient Clipping (The Optimization Fix)
*   **Theory:** The empirical observation that the network flips from black to white between `ETA = 0.40` and `ETA = 0.50` proves that the localized step sizes are causing violent over-correction.
*   **Implementation Strategy:**
    1.  **ETA Capping:** The `ETA_TARGET` ramp schedule is too aggressive. It must be capped at a much lower threshold (e.g., maximum `0.10` or `0.20`) where the network was historically stable and generating "muddy grids".
    2.  **Gradient Clipping on Target Calculation:** Apply `tf.clip_by_norm` directly to the local derivative *before* it multiplies with `ETA`: `T = h - ETA * tf.clip_by_norm(grad_h, CLIP_NORM)`. This guarantees that even if the top-level discriminator requests a massive color inversion, the hidden layers are strictly bounded in how far they can jump per epoch.
*   **Research Question:** Does heavily clipping localized gradients permanently restrict the network's capacity to learn the high-frequency structural details required for the complex Anime Face dataset?