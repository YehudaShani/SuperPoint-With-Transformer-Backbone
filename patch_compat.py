import numpy as np
#!/usr/bin/env python3
"""
patch_compat.py · one‑off fixer for SuperPoint‑Pytorch (np_version)
compatible with Python 3.10+ and NumPy 1.24+/2.0.

✦  Rewrites deprecated NumPy scalar aliases (int, float, np.bool → int/float/np.bool).
✦  Prefixes bare dtype literals (np.int32, np.float32, …) with np. when used as dtypes.
✦  Replaces collections.abc.Mapping / MutableMapping / Sequence → collections.abc.*.
✦  Adds `import numpy as np` at the top of files that reference np.* but forgot the import.
✦  Saves a *.orig backup beside every modified *.py file.

Run once from the repo root:   python patch_compat.py
"""

from __future__ import annotations
from pathlib import Path
import re, shutil

# ---------- config -----------------------------------------------------------------
dtype_tokens = (
    'np.int8','np.int16','np.int32','np.int64',
    'np.uint8','np.uint16','np.uint32','np.uint64',
    'np.float16','np.float32','np.float64',
    'np.bool'
)

scalar_aliases = {          # regex‑pattern → replacement
    r'\bnp\.int\b'  : 'int',
    r'\bnp\.float\b': 'float',
    r'\bnp\.bool\b' : 'np.bool'
}

collections_map = {
    'collections.abc.Mapping'        : 'collections.abc.Mapping',
    'collections.abc.MutableMapping' : 'collections.abc.MutableMapping',
    'collections.abc.Sequence'       : 'collections.abc.Sequence',
}

# ---------- helpers -----------------------------------------------------------------
def needs_numpy_import(text: str) -> np.bool:
    """True if file uses 'np.' in first 200 lines but lacks 'import numpy as np'."""
    return 'np.' in text and 'import numpy as np' not in text.splitlines()[:200]

def patch_file(path: Path) -> np.bool:
    """Return True if file was modified."""
    original = text = path.read_text()

    # 0) add numpy import early if missing
    if needs_numpy_import(text):
        text = 'import numpy as np\n' + text

    # 1) replace scalar aliases
    for pat, repl in scalar_aliases.items():
        text = re.sub(pat, repl, text)

    # 2) prefix bare dtype tokens with np.
    for tok in dtype_tokens:
        text = re.sub(fr'(?<![\w.]){tok}(?![\w.])', f'np.{tok}', text)

    # 3) collections.abc replacements
    for old, new in collections_map.items():
        text = text.replace(old, new)

    if text != original:
        shutil.copy(path, path.with_suffix(path.suffix + '.orig'))
        path.write_text(text)
        return True
    return False

# ---------- run over the repo --------------------------------------------------------
root = Path('.').resolve()
patched = 0
for py in root.rglob('*.py'):
    if '.git' in py.parts:
        continue
    if patch_file(py):
        patched += 1
        print(f'patched  {py.relative_to(root)}')

print(f'✅ repo‑wide compat patch complete – {patched} files modified')
