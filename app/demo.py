"""Gradio web demo — binary sentiment classification"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import gradio as gr
import pandas as pd
from app.model_loader import SentimentPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results.csv"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
TRAIN_CSV = str(PROJECT_ROOT / "data" / "processed" / "train.csv")

# ---- Read results ----
df = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None
if df is not None:
    df = df.sort_values("macro_f1", ascending=False)
    MODEL_STATS = {
        row["model"]: {"macro_f1": row["macro_f1"], "accuracy": row["accuracy"]}
        for _, row in df.iterrows()
    }
    BEST_MODEL = df.iloc[0]["model"]
else:
    MODEL_STATS = {}
    BEST_MODEL = "roberta"

# ---- Model registry ----
# (model_type, checkpoint_path, needs_vectorizer, needs_word2idx)
MODEL_REGISTRY = {
    "roberta":       ("roberta",     CKPT_DIR / "roberta_best"),
    "bert":          ("bert",        CKPT_DIR / "bert_best"),
    "bigru_attn":    ("bigru_attn",  CKPT_DIR / "bigru_attn_best.pth"),
    "textcnn":       ("textcnn",     CKPT_DIR / "textcnn_best.pth"),
    "LinearSVC":     ("svm",         CKPT_DIR / "linearsvc.pkl"),
    "MultinomialNB": ("naive_bayes", CKPT_DIR / "multinomialnb.pkl"),
    "XGBoost":       ("xgboost",     CKPT_DIR / "xgboost.json"),
}
AVAILABLE_MODELS = [m for m in MODEL_REGISTRY if m in MODEL_STATS]

# ---- Shared resources (lazy) ----
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
        print("Building vocabulary (one-time, ~9s)...")
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


def predict(text, model_name):
    if not text or not text.strip():
        return "请输入评论内容", {"差评": 0.5, "好评": 0.5}
    p = get_predictor(model_name)
    label, probs = p.predict(text.strip())
    return f"**{label}**（差评 {probs[0]:.2%} / 好评 {probs[1]:.2%}）", {"差评": float(probs[0]), "好评": float(probs[1])}


def switch_model(model_name):
    stats = MODEL_STATS.get(model_name, {})
    info = f"Macro-F1: **{stats.get('macro_f1', 0):.4f}** | Accuracy: **{stats.get('accuracy', 0):.4f}**"
    get_predictor(model_name)  # warm up
    return info


# ---- UI ----
with gr.Blocks(title="电商评论情感分析") as demo:
    gr.Markdown("# 电商评论情感二分类 Demo")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=AVAILABLE_MODELS, value=BEST_MODEL,
            label="选择模型", interactive=True)
        model_info = gr.Markdown(
            f"Macro-F1: **{MODEL_STATS[BEST_MODEL]['macro_f1']:.4f}** | "
            f"Accuracy: **{MODEL_STATS[BEST_MODEL]['accuracy']:.4f}**")

    inp = gr.Textbox(lines=4, placeholder="在此输入中文电商评论...", label="评论内容")
    btn = gr.Button("分析情感", variant="primary")

    with gr.Row():
        lbl = gr.Label(label="情感概率", num_top_classes=2)
        result = gr.Markdown()

    gr.Examples(
        examples=[
            "产品质量很好，做工精细，物流也很快，非常满意！",
            "用了不到一个月就坏了，客服也不理人，太失望了。",
            "一般般吧，没想象中那么好用，凑合着用。",
        ],
        inputs=inp,
    )

    btn.click(fn=predict, inputs=[inp, model_dropdown], outputs=[result, lbl])
    model_dropdown.change(fn=switch_model, inputs=model_dropdown, outputs=model_info)

if __name__ == "__main__":
    demo.launch()
