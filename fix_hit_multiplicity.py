"""
Run this once from your project root to patch the false-positive bug in checker.py.

    PYTHONPATH=. python fix_hit_multiplicity.py

What it fixes:
    The _check_hit_multiplicity rule was flagging rows where energy < threshold
    AND hit_multiplicity > 0. This generates ~1,907 false positives on clean data
    because low-energy events legitimately register hits.

    Only the meaningful violation is kept:
        above threshold with zero hits (energy deposited but no hit recorded).
"""

from pathlib import Path

CHECKER = Path("rules/checker.py")

OLD = '        bad = np.where((above & (mult == 0)) | (~above & (mult != 0)))[0]'
NEW = '        bad = np.where(above & (mult == 0))[0]  # only: above threshold, zero hits'

src = CHECKER.read_text()
if OLD not in src:
    print("ERROR: expected line not found — checker.py may have already been patched")
    print(f"Looking for:\n  {OLD}")
else:
    patched = src.replace(OLD, NEW)
    CHECKER.write_text(patched)
    print("Patched rules/checker.py — removed false-positive condition.")
    print("Re-run: PYTHONPATH=. python rules/checker.py")
    print("Then:   PYTHONPATH=. python eval/compare.py")