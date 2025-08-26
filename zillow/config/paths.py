from pathlib import Path
from dotenv import load_dotenv

PROJ_ROOT = Path(__file__).resolve().parents[2]
#print(f"Project root directory: {PROJ_ROOT}")

load_dotenv(PROJ_ROOT / ".env")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ANALYSIS_RESULTS_DIR = REPORTS_DIR / "analysis_results"
DRIFTS_DIR = ANALYSIS_RESULTS_DIR / "drifts"

SCRIPTS_DIR = PROJ_ROOT / "scripts"

CONFIG_PATH = PROJ_ROOT / "configs"

