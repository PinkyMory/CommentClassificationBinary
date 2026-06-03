from pathlib import Path

# ---- Project root (auto-inferred, not hardcoded) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Data paths ----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# ---- Output paths ----
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "results.csv"

# ---- Random seed ----
SEED = 42

# ---- Sampling params ----
USE_ALL_DATA = True
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
USE_CLASS_WEIGHT = True

# ---- Text preprocessing ----
MAX_SEQ_LEN = 128
MAX_SEQ_LEN_BERT = 256

# ---- TF-IDF ----
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ---- Word2Vec ----
EMBEDDING_DIM = 300

# ---- Deep learning from scratch ----
BATCH_SIZE_DL = 64
EPOCHS_DL = 30
LR_DL = 1e-3
DROPOUT = 0.5

# ---- Pretrained models ----
BERT_MODEL_NAME = "bert-base-chinese"
ROBERTA_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
BATCH_SIZE_PRETRAINED = 16
EPOCHS_PRETRAINED = 5
LR_PRETRAINED = 2e-5

# ---- Label mapping (binary) ----
LABEL_MAP = {0: "差评", 1: "好评"}
NUM_CLASSES = 2

# ---- Star-to-label mapping (binary: star<=3 as negative) ----
def star_to_label(star: int) -> int:
    """1-3 star -> negative(0), 4-5 star -> positive(1)"""
    if star <= 3:
        return 0
    else:
        return 1
