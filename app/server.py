"""
FastAPI 后端服务
=============================================================================
为前端 HTML 页面提供 REST API 接口，替代原 Gradio 界面。

启动方式：
  python app/server.py
  然后访问 http://127.0.0.1:8000

API 端点：
  GET  /              → 前端 HTML 页面
  GET  /api/models    → 可用模型列表及指标
  POST /api/predict   → 情感预测
  POST /api/preload   → 预热加载模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model_loader import SentimentPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results.csv"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
TRAIN_CSV = str(PROJECT_ROOT / "data" / "processed" / "train.csv")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ============================================================================
# 模型注册表
# ============================================================================
MODEL_REGISTRY = {
    "roberta":       ("roberta",     CKPT_DIR / "roberta_best"),
    "bert":          ("bert",        CKPT_DIR / "bert_best"),
    "bigru_attn":    ("bigru_attn",  CKPT_DIR / "bigru_attn_best.pth"),
    "textcnn":       ("textcnn",     CKPT_DIR / "textcnn_best.pth"),
    "LinearSVC":     ("svm",         CKPT_DIR / "linearsvc.pkl"),
    "MultinomialNB": ("naive_bayes", CKPT_DIR / "multinomialnb.pkl"),
    "XGBoost":       ("xgboost",     CKPT_DIR / "xgboost.json"),
}

# ============================================================================
# 加载模型指标
# ============================================================================
df = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None
if df is not None:
    df = df.sort_values("macro_f1", ascending=False)
    MODEL_STATS = {
        row["model"]: {"macro_f1": round(row["macro_f1"], 4), "accuracy": round(row["accuracy"], 4)}
        for _, row in df.iterrows()
    }
else:
    MODEL_STATS = {}

AVAILABLE_MODELS = [m for m in MODEL_REGISTRY if m in MODEL_STATS]

# ============================================================================
# 延迟加载缓存
# ============================================================================
_predictors = {}
_vectorizer = None
_word2idx = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        vp = CKPT_DIR / "tfidf_vectorizer.pkl"
        with open(vp, "rb") as f:
            _vectorizer = pickle.load(f)
    return _vectorizer


def get_word2idx():
    global _word2idx
    if _word2idx is None:
        from src.dataset import build_vocab_from_csv
        _word2idx = build_vocab_from_csv(TRAIN_CSV, min_freq=2)
    return _word2idx


def get_predictor(model_name: str):
    if model_name not in _predictors:
        mtype, mpath = MODEL_REGISTRY[model_name]
        kwargs = {}
        if mtype in ("svm", "naive_bayes"):
            kwargs["vectorizer"] = get_vectorizer()
        elif mtype in ("textcnn", "bigru_attn"):
            kwargs["word2idx"] = get_word2idx()
        _predictors[model_name] = SentimentPredictor(mtype, str(mpath), **kwargs)
    return _predictors[model_name]


# ============================================================================
# FastAPI 应用
# ============================================================================
app = FastAPI(title="电商评论情感分析", version="1.0")


@app.get("/api/models")
async def list_models():
    """返回可用模型及其指标"""
    return {
        "models": [
            {
                "name": m,
                "macro_f1": MODEL_STATS.get(m, {}).get("macro_f1", 0),
                "accuracy": MODEL_STATS.get(m, {}).get("accuracy", 0),
            }
            for m in AVAILABLE_MODELS
        ],
        "best_model": AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "roberta",
    }


class PredictRequest(BaseModel):
    text: str
    model_name: str


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """情感预测"""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")

    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"未知模型: {req.model_name}")

    p = get_predictor(req.model_name)
    label, probs = p.predict(text)
    return {
        "label": label,
        "negative_prob": round(float(probs[0]), 4),
        "positive_prob": round(float(probs[1]), 4),
    }


class PreloadRequest(BaseModel):
    model_name: str


@app.post("/api/preload")
async def preload(req: PreloadRequest):
    """预热加载模型"""
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"未知模型: {req.model_name}")
    get_predictor(req.model_name)
    return {"status": "ok", "model": req.model_name}


# ============================================================================
# 静态文件 & 前端页面
# ============================================================================
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
