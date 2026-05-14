import json

with open('phase4-pcgan-localizedadamw.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])

        # We need to drop LAMBDA_INV and fix the learning rate.
        # Right now the generator optimizer uses weight decay but in GANs that can kill the generator.
        if "LAMBDA_INV = tf.constant(1.0" in source:
            source = source.replace("LAMBDA_INV = tf.constant(1.0, dtype=tf.float32)", "LAMBDA_INV = tf.constant(0.1, dtype=tf.float32)")

        cell['source'] = [line + '\n' for line in source.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('phase4-pcgan-localizedadamw.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
