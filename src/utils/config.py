from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts'
DB_PATH = ARTIFACT_DIR / 'policyguard.db'
