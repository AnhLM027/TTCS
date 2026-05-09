import json
import sys
for file in sys.argv[1:]:
    print(f'\n\n# FILE: {file}\n')
    with open(file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            print(''.join(cell.get('source', [])))
        elif cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            print('\n```python\n' + source + '\n```\n')
