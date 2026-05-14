import json

with open('/tmp/notebook.json', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'train_ds' in source:
            print(f"--- CELL {i} ---")
            print(source)
            print("-----------------")
