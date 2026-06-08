"""
Gradio Web 演示界面
=============================================================================
基于 Gradio Blocks 构建的情感分析网页应用。

功能：
  - 模型选择下拉框（自动显示各模型的 Macro-F1 和 Accuracy）
  - 文本输入框 + 情感分析按钮
  - 预测结果展示（情感标签 + 概率分布）
  - 示例输入（点击自动填入）
  - 延迟加载预测器（首次选择某模型时才加载，节省内存）

架构：
  demo.py → model_loader.py → 各类模型权重文件
                              → src/ 模块（数据预处理、模型定义等）

支持的模型（7种）：
  预训练模型:     roberta, bert
  深度学习模型:   bigru_attn, textcnn
  传统机器学习:   LinearSVC, MultinomialNB, XGBoost

使用方法：
  python app/demo.py
  然后访问 http://127.0.0.1:7860
"""

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

# ============================================================================
# 读取模型指标数据
# ============================================================================
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


# ============================================================================
# HTML 模板
# ============================================================================

def build_result_html(label: str, probs) -> str:
    """构建富文本结果卡片"""
    neg_pct = round(float(probs[0]) * 100, 1)
    pos_pct = round(float(probs[1]) * 100, 1)

    if label == "好评":
        emoji = "&#x1F60A;"
        sentiment_en = "POSITIVE"
        accent = "#10b981"
        accent_soft = "#d1fae5"
        bg_gradient = "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"
    else:
        emoji = "&#x1F61E;"
        sentiment_en = "NEGATIVE"
        accent = "#ef4444"
        accent_soft = "#fee2e2"
        bg_gradient = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"

    return f"""
    <div style="
        background: {bg_gradient};
        border: 1px solid {accent_soft};
        border-radius: 20px;
        padding: 32px 28px 24px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
    ">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;">
            <div style="
                font-size:48px;
                line-height:1;
                filter:drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            ">{emoji}</div>
            <div>
                <div style="
                    font-size:28px;
                    font-weight:700;
                    color:#1e293b;
                    line-height:1.2;
                ">{label}</div>
                <div style="
                    font-size:12px;
                    font-weight:600;
                    letter-spacing:2px;
                    color:{accent};
                    text-transform:uppercase;
                ">{sentiment_en}</div>
            </div>
        </div>

        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px;color:#475569;">
                <span style="font-weight:600;">&#x1F44D; 好评</span>
                <span style="font-weight:700;color:{accent if label == '好评' else '#94a3b8'};">{pos_pct}%</span>
            </div>
            <div style="
                height:10px;
                background:#e2e8f0;
                border-radius:99px;
                overflow:hidden;
                box-shadow:inset 0 1px 3px rgba(0,0,0,0.08);
            ">
                <div style="
                    width:{pos_pct}%;
                    height:100%;
                    background:linear-gradient(90deg, #34d399, #10b981);
                    border-radius:99px;
                    transition:width 0.6s cubic-bezier(0.4,0,0.2,1);
                "></div>
            </div>
        </div>

        <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px;color:#475569;">
                <span style="font-weight:600;">&#x1F44E; 差评</span>
                <span style="font-weight:700;color:{accent if label == '差评' else '#94a3b8'};">{neg_pct}%</span>
            </div>
            <div style="
                height:10px;
                background:#e2e8f0;
                border-radius:99px;
                overflow:hidden;
                box-shadow:inset 0 1px 3px rgba(0,0,0,0.08);
            ">
                <div style="
                    width:{neg_pct}%;
                    height:100%;
                    background:linear-gradient(90deg, #f87171, #ef4444);
                    border-radius:99px;
                    transition:width 0.6s cubic-bezier(0.4,0,0.2,1);
                "></div>
            </div>
        </div>
    </div>"""


def build_model_stats_html(model_name: str) -> str:
    """构建模型指标徽章"""
    stats = MODEL_STATS.get(model_name, {})
    macro_f1 = stats.get('macro_f1', 0)
    accuracy = stats.get('accuracy', 0)

    def badge(value, label, color):
        return f"""
        <span style="
            display:inline-flex;align-items:center;gap:6px;
            background:{color}15;border:1px solid {color}30;
            border-radius:99px;padding:6px 14px;
            font-size:13px;font-weight:500;color:{color};
        ">
            <span style="font-weight:700;font-size:15px;">{value:.4f}</span>
            <span style="opacity:0.75;">{label}</span>
        </span>"""

    return f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        {badge(macro_f1, "Macro-F1", "#6366f1")}
        {badge(accuracy, "Accuracy", "#0ea5e9")}
    </div>"""


# ============================================================================
# 界面回调函数
# ============================================================================

def predict(text, model_name):
    if not text or not text.strip():
        return (
            """<div style="
                padding:40px 20px;text-align:center;color:#94a3b8;
                font-size:15px;border:2px dashed #e2e8f0;
                border-radius:16px;margin:8px 0;
            ">&#128220; 请在输入框中输入评论内容后点击分析</div>""",
            {"差评": 0.5, "好评": 0.5}
        )

    p = get_predictor(model_name)
    label, probs = p.predict(text.strip())
    return (
        build_result_html(label, probs),
        {"差评": float(probs[0]), "好评": float(probs[1])}
    )


def switch_model(model_name):
    stats = MODEL_STATS.get(model_name, {})
    get_predictor(model_name)
    return build_model_stats_html(model_name)


# ============================================================================
# 自定义 CSS
# ============================================================================

CUSTOM_CSS = """
/* ---- 全局 ---- */
.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
}

/* ---- 页头渐变 ---- */
.main-header {
    text-align: center;
    padding: 36px 20px 28px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 24px;
    margin-bottom: 8px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.25);
}
.main-header h1 {
    color: #fff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px !important;
    letter-spacing: -0.02em;
}
.main-header p {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ---- 卡片容器 ---- */
.section-card {
    background: #fff;
    border: 1px solid #f1f5f9;
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ---- Dropdown 美化 ---- */
.model-dropdown input, .model-dropdown button {
    font-size: 15px !important;
    font-weight: 500 !important;
}
.model-dropdown label {
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 6px !important;
}

/* ---- 输入框美化 ---- */
.text-input textarea {
    font-size: 15px !important;
    line-height: 1.7 !important;
    border-radius: 14px !important;
    border-color: #e2e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 14px 16px !important;
}
.text-input textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ---- 按钮美化 ---- */
.analyze-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.01em;
}
.analyze-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}
.analyze-btn:active {
    transform: translateY(0) !important;
}

/* ---- Label 组件美化 ---- */
.output-label {
    border-radius: 16px !important;
    overflow: hidden;
}

/* ---- Examples 区域 ---- */
.examples-section {
    margin-top: 8px;
}
.examples-section .examples-title {
    font-weight: 600;
    color: #475569;
    font-size: 14px;
}

/* ---- 响应式 ---- */
@media (max-width: 640px) {
    .gradio-container {
        padding: 12px !important;
    }
    .main-header {
        padding: 24px 16px 20px;
        border-radius: 18px;
    }
    .main-header h1 {
        font-size: 1.5rem !important;
    }
}
"""

# ============================================================================
# Gradio UI 构建
# ============================================================================

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="lg",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="电商评论情感分析") as demo:

    # ---- 页头 ----
    gr.HTML(f"""
    <div class="main-header">
        <h1>&#128270; 电商评论情感分析</h1>
        <p>Binary Sentiment Classification &middot; {len(AVAILABLE_MODELS)} 个模型可用</p>
    </div>
    """)

    # ---- 模型选择区 ----
    with gr.Group(elem_classes="section-card"):
        gr.Markdown("### &#9881;&#65039; 模型选择")
        with gr.Row(equal_height=True):
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=BEST_MODEL,
                label="选择模型",
                interactive=True,
                elem_classes="model-dropdown",
                scale=2,
            )
            with gr.Column(min_width=180, scale=1):
                model_info = gr.HTML(
                    build_model_stats_html(BEST_MODEL),
                    show_label=False,
                )

    # ---- 输入区 ----
    with gr.Group(elem_classes="section-card"):
        inp = gr.Textbox(
            lines=4,
            placeholder="&#128172; 在此输入中文电商评论，例如：\"产品质量很好，物流也很快，非常满意！\"",
            label="评论内容",
            elem_classes="text-input",
        )
        btn = gr.Button(
            "&#128269; 分析情感",
            variant="primary",
            elem_classes="analyze-btn",
        )

    # ---- 结果区 ----
    with gr.Group(elem_classes="section-card"):
        gr.Markdown("### &#128202; 分析结果")
        with gr.Row(equal_height=False):
            result_html = gr.HTML(
                value="""<div style="
                    padding:40px 20px;text-align:center;color:#94a3b8;
                    font-size:15px;border:2px dashed #e2e8f0;
                    border-radius:16px;
                ">&#127775; 等待输入评论进行分析...</div>""",
                show_label=False,
                scale=3,
            )
            lbl = gr.Label(
                label="概率分布",
                num_top_classes=2,
                scale=2,
                elem_classes="output-label",
            )

    # ---- 示例区 ----
    with gr.Group(elem_classes="section-card"):
        gr.Markdown("### &#128161; 试试这些例子")
        gr.Examples(
            examples=[
                "产品质量很好，做工精细，物流也很快，非常满意！",
                "用了不到一个月就坏了，客服也不理人，太失望了。",
                "一般般吧，没想象中那么好用，凑合着用。",
            ],
            inputs=inp,
        )

    # ---- 事件绑定 ----
    btn.click(fn=predict, inputs=[inp, model_dropdown], outputs=[result_html, lbl])
    model_dropdown.change(fn=switch_model, inputs=model_dropdown, outputs=model_info)

if __name__ == "__main__":
    demo.launch(
        theme=theme,
        css=CUSTOM_CSS,
    )
