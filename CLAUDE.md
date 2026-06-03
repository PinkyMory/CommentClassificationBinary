# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the conda environment `llm` at `D:\develop\Anaconda\envs\llm`. Python 3.12, torch 2.11+cu126, transformers 5.9. All dependencies from `requirements.txt` are installed here.

```bash
conda activate llm
python -c "import torch; print(torch.cuda.is_available())"  # Should print True on RTX 4060
```

## Architecture

**Binary sentiment classification** (negative vs positive). Stars 1-3 are treated as negative, stars 4-5 as positive.

**Shared library pattern**: `src/` is the shared module. Every script in `scripts/` starts with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

This ensures scripts work from any working directory without package installation. All config paths in `src/config.py` use `Path(__file__).resolve().parent.parent` to derive `PROJECT_ROOT`.

**Training scripts are independent entry points**. They do not import each other. They share `src/metrics.py` (evaluation) and `src/plot.py` (visualization), but each owns its own `train_model()` function and its own model definitions.

**Data flow**: `01_sampling.py` (excludes star=3) → `train.csv / val.csv / test.csv` → `02/03/04_train_*.py` each read these CSVs independently → append results to `outputs/results.csv` → `05_evaluate_all.py` reads `results.csv` for comparison charts.

## Class imbalance strategy

With star<=3 as negative, the dataset has ~19% negative, ~81% positive. Three different strategies per layer:

- **Traditional ML (02)**: `LinearSVC(class_weight='balanced')`, XGBoost with `compute_sample_weight('balanced')`, MultinomialNB has no weight (relies on naive prior).
- **DL from scratch (03)**: `WeightedRandomSampler` in DataLoader + weighted `CrossEntropyLoss` in `train_loop`.
- **Pretrained (04)**: Custom `WeightedTrainer` subclass overriding `compute_loss` with weighted `CrossEntropyLoss`.

Primary metric is **macro-F1**, not accuracy (which is misleading under imbalance).

## HuggingFace API changes (transformers 5.x)

The code uses transformers 5.x conventions:
- `eval_strategy` (not deprecated `evaluation_strategy`)
- `processing_class` in Trainer (not deprecated `tokenizer`)
- `warmup_steps` computed from dataset size (not deprecated `warmup_ratio`)

## Pretrained model caching

`04_train_pretrained.py` uses a `load_model_and_tokenizer()` helper that tries `local_files_only=True` first, then falls back to downloading. This avoids timeout errors on subsequent runs when models are already cached.

## Label mapping (binary)

| Star | Label |
|------|-------|
| 1-3  | 0 (差评 / negative) |
| 4-5  | 1 (好评 / positive) |

## Dataset quirks

- `TokenizedDataset` (used by 03) pre-tokenizes all samples in `__init__`. Do NOT move `jieba.cut()` back into `__getitem__` — it would re-tokenize every epoch.
- `build_embedding_matrix` accepts `wv_path=None` and falls back to random init. This lets `03_train_dl.py` run without pre-downloaded word vectors.
- The raw dataset's text column is `评论内容`, star column is `评分`. Column detection in `01_sampling.py` uses priority-ordered candidate matching.
