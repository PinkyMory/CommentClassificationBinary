"""Gradio web demo (binary sentiment)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd
from app.model_loader import SentimentPredictor


def _find_best_model():
    results_path = Path(__file__).resolve().parent.parent / "outputs" / "results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        best = df.loc[df["macro_f1"].idxmax()]
        print(f"Auto-selected best model: {best['model']} (Macro-F1: {best['macro_f1']:.4f})")
        return best["model"]
    return None


BEST_MODEL_NAME = _find_best_model()

MODEL_CONFIGS = {
    "textcnn":     {"type": "textcnn",     "path": "checkpoints/textcnn_best.pth"},
    "bigru_attn":  {"type": "bigru_attn",  "path": "checkpoints/bigru_attn_best.pth"},
    "bert":        {"type": "bert",        "path": "checkpoints/bert_best"},
    "roberta":     {"type": "roberta",     "path": "checkpoints/roberta_best"},
}

predictor = None


def load_model(model_name: str):
    global predictor
    cfg = MODEL_CONFIGS[model_name]
    if cfg["type"] in ("bert", "roberta"):
        predictor = SentimentPredictor(cfg["type"], cfg["path"])
        return f"Loaded {model_name}"
    else:
        return f"{model_name} requires vectorizer/word2idx; run the corresponding training script first"


def classify(text: str):
    if predictor is None:
        return {"Please select a model first": 1.0}
    label, probs = predictor.predict(text)
    return {predictor.labels[i]: float(p) for i, p in enumerate(probs)}


if BEST_MODEL_NAME:
    load_model(BEST_MODEL_NAME)

model_choices = list(MODEL_CONFIGS.keys())
default_model = BEST_MODEL_NAME if BEST_MODEL_NAME in model_choices else model_choices[0]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=5, placeholder="Enter a JD product review here...", label="Review Text"),
    outputs=gr.Label(num_top_classes=2, label="Sentiment Prediction"),
    title="JD Product Review Sentiment Binary Classification Demo",
    description=f"Current model: {BEST_MODEL_NAME or default_model}",
    examples=[
        ["Great quality, very satisfied after using it for a while!"],
        ["Terrible, broke after two days, worst purchase ever!"],
        ["Delivery was fast, packaging is good, product works well."],
    ],
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
