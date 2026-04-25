import sys
from pathlib import Path

_POD = Path(__file__).resolve().parent.parent / "pod"
if str(_POD) not in sys.path:
    sys.path.insert(0, str(_POD))
