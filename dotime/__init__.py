"""Do-Over-Time-PFN: In-context causal effect estimation for temporal data."""

import sys
import os

# Add CTP and TempoPFN to path for imports
_CTP_ROOT = os.environ.get("CTP_ROOT", "/home/dennis/repos/ctp")
for _path in [_CTP_ROOT, os.path.join(_CTP_ROOT, "TempoPFN")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
