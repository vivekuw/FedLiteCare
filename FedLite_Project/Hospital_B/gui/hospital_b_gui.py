"""Legacy compatibility wrapper for the Hospital B GUI launcher."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
for candidate in (CURRENT_FILE.parent, *CURRENT_FILE.parents):
    package_parent = candidate / "FedLite_Project"
    if package_parent.is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from FedLite_Project.Hospital_B.hospital_b_app import main


if __name__ == "__main__":
    main()
