import json

with open('phase4-pcgan-localizedadamw.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])

        # In DTP, the backward mapping must be trained to invert the forward mapping.
        # Right now we have:
        # T3 = tf.stop_gradient(generator.block4.predict_back_v2(T4) + h3 - pred4)
        # where pred4 = generator.block4.predict_back_v2(h4) (computed in forward pass)
        #
        # But wait, looking at the user's logs:
        # "Total params: 4,317,124 ... Trainable params: 2,766,657 ... Non-trainable params: 1,550,467"
        # The discriminator has non-trainable parameters! That's `self.up_conv` which was frozen to False in the previous step.
        #
        # In the generator block `dtp_block_g`:
        # `upscaled = self.upsample(x)` (nearest upsampling)
        # `features = self.conv_up(upscaled)`
        # `h = self.relu(self.ln(features))`
        # `prediction = self.predict_back_v2(h)`
        # `predict_back_v2` is Conv2D with stride 2. This means `prediction` is the same size as `x`.

        # Wait, the inverse loss formula is currently:
        # noisy_h = h_out + noise
        # pred_x = predict_back_v2(noisy_h)
        # loss = MSE(pred_x, clean_in)

        # But `h_out` is passed into `predict_back_v2` WITHOUT the activation of the previous layer, it maps back to `x_in`.
        # However, `x_in` to `block4` is `h3`. So `MSE(pred_x, in4)` minimizes `MSE(predict_back_v2(h4), h3)`. This is correct.

        # But why is the generator loss stuck at ~9.5 and producing blue grids?
        # Because we are using 1.0 * inv_loss_b4! The inverse loss might be completely dominating the target loss,
        # preventing the generator from actually following T4, T3, T2, T1.
        # In fact, LAMBDA_INV = 1.0 is huge for an MSE pixel/feature space.
        # We need to scale down the target loss or scale up the inverse loss, OR we are adding them incorrectly.

        # Let's look at the guide again.
        # "The Stability Correction: ... \tau_{l-1} = h_{l-1} + g_l(\tau_l) - g_l(h_l)"
        # T3 = generator.block4.predict_back_v2(T4) + h3 - generator.block4.predict_back_v2(h4)
        # This matches what we have: T3 = predict_back_v2(T4) + h3 - pred4.

        if "LAMBDA_INV = tf.constant(1.0, dtype=tf.float32)      # Weight for inverse mapping loss" in source:
            # Let's change this to 0.1 to prevent it from dominating the generator's adversarial objective.
            pass
