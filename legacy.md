# Legacy Implementation: Manual Localized AdamW

## Context
During the early phases of this project (before the transition to Difference Target Propagation / DTP), the architecture attempted to achieve "localized learning" by intercepting global gradients and applying manual weight updates block-by-block.

This file preserves the manual implementation of the AdamW optimizer that was used in `train_step_phase4` before it was replaced by standard independent Keras optimizer instances.

## Why this was retired
While this manual math successfully implemented AdamW (complete with bias correction and decoupled weight decay), it was ultimately a "Post-Backprop Illusion." Even though the optimizer updates were unrolled manually, the gradients themselves were still derived via global backpropagation through the entire network using the chain rule.

With the shift to true DTP, the global graph is severed, and each layer receives its own specific target tensor. Because learning is now truly localized at the block level, we can simply instantiate independent `tf.keras.optimizers.AdamW` instances for each block, making this manual boilerplate obsolete.

## The Legacy Code

Below is the manual AdamW implementation, preserving the logic for momentum tracking, velocity tracking, step increments, and bias correction.

```python
import tensorflow as tf

# 1. LOCALIZED ADAM-W HYPERPARAMETERS
MANUAL_LR = tf.constant(2e-5, dtype=tf.float32)
BETA_1 = tf.constant(0.9, dtype=tf.float32)
BETA_2 = tf.constant(0.999, dtype=tf.float32)
EPSILON = tf.constant(1e-7, dtype=tf.float32)
WEIGHT_DECAY = tf.constant(1e-4, dtype=tf.float32)
CLIP_NORM = tf.constant(1.0, dtype=tf.float32)

# 2. INITIALIZE LOCAL MEMORY
generator_momentums = [tf.Variable(tf.zeros_like(var), trainable=False) for var in generator.trainable_variables]
generator_velocities = [tf.Variable(tf.zeros_like(var), trainable=False) for var in generator.trainable_variables]

# Initialize step counters for AdamW bias correction
gen_step = tf.Variable(0, trainable=False, dtype=tf.float32)

@tf.function
def train_step_phase4(real_images):
    # ... (Forward pass and loss calculation) ...

    # CALCULATE GRADIENTS
    gen_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, CLIP_NORM)

    # Increment step counter
    gen_step.assign_add(1.0)

    # Bias correction factors
    gen_bias_correction1 = 1.0 - tf.pow(BETA_1, gen_step)
    gen_bias_correction2 = 1.0 - tf.pow(BETA_2, gen_step)

    # MANUAL ADAM-W UPDATE LOOP (GENERATOR)
    for i, (grad, var) in enumerate(zip(gen_gradients, generator.trainable_variables)):
        if grad is not None:
            m = generator_momentums[i]
            v = generator_velocities[i]

            # Update biased first and second moments
            m.assign(BETA_1 * m + (1.0 - BETA_1) * grad)
            v.assign(BETA_2 * v + (1.0 - BETA_2) * tf.square(grad))

            # Apply bias correction
            m_hat = m / gen_bias_correction1
            v_hat = v / gen_bias_correction2

            # Calculate final update step with decoupled weight decay
            update = (MANUAL_LR * m_hat / (tf.sqrt(v_hat) + EPSILON)) + (WEIGHT_DECAY * MANUAL_LR * var)
            var.assign_sub(update)
```