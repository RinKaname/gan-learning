# Phase 7 Research Notes: Target Escalation & Feedback Oscillation

## Observation
As of Epoch 10, the DTP-GAN has shifted from generating complex "muddy grid" patterns (Epochs 1-5) into a **Whiteout Collapse** (pure white output). This follows a brief period of **Blackout Collapse** around Epoch 6.

## Diagnosis: Target Escalation & Oscillation
While the structural safety nets implemented in Phase 6 (Top-Level Target Clamping, Label Smoothing, TTUR) successfully prevented explosive `NaN` corruption, the network is now suffering from a deeper systemic issue inherent to Difference Target Propagation (DTP): **Target Escalation**.

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

### 3. Localized Target Gradient Clipping (The Optimization Fix)
*   **Theory:** Even with clamped targets and accurate inverses, the calculated local gradient (`grad_h = tape.gradient(local_loss, h)`) can still spike aggressively if the requested target `T` is drastically different from the current activation `h`.
*   **Implementation Strategy:** Apply local gradient clipping specifically to the derivative before updating the local target calculation: `T = h - ETA * tf.clip_by_norm(grad_h, CLIP_NORM)`. This guarantees that the generator takes smooth, bounded steps toward the requested representation per iteration, mitigating severe oscillations.
*   **Research Question:** Does localized gradient clipping restrict the network's ability to learn complex, high-frequency features in standard DTP setups?