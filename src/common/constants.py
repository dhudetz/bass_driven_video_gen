from pathlib import Path

# ========== Directories ===========
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
SRC_DIR = ROOT_DIR / 'src'

DEFAULT_CONFIG_PATH = CONFIG_DIR / "saved_constants.txt"