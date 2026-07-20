# Path migration notes (V3 active)

## Active development lives here

Put V19+ code under `v3/source_versions/` (e.g. `v3/source_versions/v19_...`).  
Train outputs → `v3/outputs/`.  
Ad-hoc scripts → `v3/scripts/`.  
Shared helpers → `v3/shared_scripts/` (includes `env_loader.py`).

## Root `.env`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared_scripts"))
from env_loader import load_root_env
load_root_env()
```

## Data

From `v3/source_versions/<pkg>/` use `../../../data` or absolute paths.
