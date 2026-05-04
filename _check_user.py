import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

path = r'c:\Users\Ujjwal\Downloads\sensorfusion_har_OPTIMIZED (4).ipynb'
nb = json.load(open(path, encoding='utf-8'))

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells'][:10]):
    src = ''.join(c['source'])
    print(f"\n=== CELL {i} [{c['cell_type']}] ===")
    print(src[:600])
