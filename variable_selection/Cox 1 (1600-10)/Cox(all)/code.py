
from pathlib import Path

files = ['file1.txt','file2.txt','file3.txt','file4.txt','file5.txt']
seen = set()
out = []
for f in files:
    p = Path(f)
    if not p.exists():
        print(f'WARN: {f} 不存在'); continue
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = line.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)

out_sorted = sorted(out, key=lambda x: (len(x), x)) 
Path('merged_unique.txt').write_text('\n'.join(out_sorted), encoding='utf-8')


