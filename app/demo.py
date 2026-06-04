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
# 词汇表重建所需的训练数据路径
TRAIN_CSV = str(PROJECT_ROOT / "data" / "processed" / "train.csv")

# ============================================================================
# 读取模型指标数据
# ============================================================================
# 从 results.csv 读取各模型的测试集指标，供前端展示
df = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None
if df is not None:
    # 按 Macro-F1 降序排列，最佳模型排最前
    df = df.sort_values("macro_f1", ascending=False)
    MODEL_STATS = {
        row["model"]: {"macro_f1": row["macro_f1"], "accuracy": row["accuracy"]}
        for _, row in df.iterrows()
    }
    # 默认选中最佳模型
    BEST_MODEL = df.iloc[0]["model"]
else:
    MODEL_STATS = {}
    BEST_MODEL = "roberta"

# ============================================================================
# 模型注册表
# ============================================================================
# (model_type, checkpoint_path)
# model_type 对应 SentimentPredictor 中的加载逻辑：
#   "roberta"/"bert"     → 预训练模型
#   "bigru_attn"/"textcnn" → 深度学习从零训练模型
#   "svm"/"naive_bayes"/"xgboost" → 传统 ML 模型
MODEL_REGISTRY = {
    "roberta":       ("roberta",     CKPT_DIR / "roberta_best"),
    "bert":          ("bert",        CKPT_DIR / "bert_best"),
    "bigru_attn":    ("bigru_attn",  CKPT_DIR / "bigru_attn_best.pth"),
    "textcnn":       ("textcnn",     CKPT_DIR / "textcnn_best.pth"),
    "LinearSVC":     ("svm",         CKPT_DIR / "linearsvc.pkl"),
    "MultinomialNB": ("naive_bayes", CKPT_DIR / "multinomialnb.pkl"),
    "XGBoost":       ("xgboost",     CKPT_DIR / "xgboost.json"),
}

# 过滤出已在 results.csv 中有记录的模型（即已训练并评估过的模型）
AVAILABLE_MODELS = [m for m in MODEL_REGISTRY if m in MODEL_STATS]

# ============================================================================
# 延迟加载缓存
# ============================================================================
# 预测器缓存：首次选择某模型时加载，后续复用，节省内存
_predictors = {}
# 共享资源（所有模型共用，只加载一次）
_vectorizer = None    # TF-IDF 向量器（传统 ML 模型用）
_word2idx = None      # 词表（DL 从零训练模型用）


def get_vectorizer():
    """加载 TF-IDF 向量器（延迟加载，只加载一次）"""
    global _vectorizer
    if _vectorizer is None:
        vp = CKPT_DIR / "tfidf_vectorizer.pkl"
        with open(vp, "rb") as f:
            _vectorizer = pickle.load(f)
    return _vectorizer


def get_word2idx():
    """构建词表（延迟加载，只构建一次，约需 9 秒）"""
    global _word2idx
    if _word2idx is None:
        from src.dataset import build_vocab_from_csv
        print("Building vocabulary (one-time, ~9s)...")
        _word2idx = build_vocab_from_csv(TRAIN_CSV, min_freq=2)
    return _word2idx


def get_predictor(model_name: str):
    """获取或创建预测器实例（延迟加载 + 缓存）

    首次请求某个模型时创建 SentimentPredictor 实例并缓存，
    后续请求直接从缓存返回，避免重复加载模型权重。

    Args:
        model_name: 模型名称（如 "roberta", "textcnn"）

    Returns:
        SentimentPredictor 实例
    """
    if model_name not in _predictors:
        mtype, mpath = MODEL_REGISTRY[model_name]
        kwargs = {}
        # 传统 ML 模型需要传入 TF-IDF 向量器
        if mtype in ("svm", "naive_bayes"):
            kwargs["vectorizer"] = get_vectorizer()
        # DL 从零训练模型需要传入词表
        elif mtype in ("textcnn", "bigru_attn"):
            kwargs["word2idx"] = get_word2idx()
        # XGBoost 和预训练模型不需要额外资源
        _predictors[model_name] = SentimentPredictor(mtype, str(mpath), **kwargs)
    return _predictors[model_name]


# ============================================================================
# 界面回调函数
# ============================================================================

def predict(text, model_name):
    """情感分析预测回调

    Args:
        text:       用户输入的评论文本
        model_name: 当前选中的模型名称

    Returns:
        (result_markdown, label_dict):
          - result_markdown: 预测结果的 Markdown 文本
          - label_dict: {"差评": prob, "好评": prob} 供 Gradio Label 组件显示
    """
    if not text or not text.strip():
        return "请输入评论内容", {"差评": 0.5, "好评": 0.5}

    p = get_predictor(model_name)
    label, probs = p.predict(text.strip())
    return (
        f"**{label}**（差评 {probs[0]:.2%} / 好评 {probs[1]:.2%}）",
        {"差评": float(probs[0]), "好评": float(probs[1])}
    )


def switch_model(model_name):
    """模型切换回调

    切换模型时：
      1. 显示新模型的指标信息
      2. 预热加载预测器（避免首次预测时等待）

    Args:
        model_name: 新选中的模型名称

    Returns:
        模型的 Macro-F1 和 Accuracy Markdown 文本
    """
    stats = MODEL_STATS.get(model_name, {})
    info = f"Macro-F1: **{stats.get('macro_f1', 0):.4f}** | Accuracy: **{stats.get('accuracy', 0):.4f}**"
    get_predictor(model_name)  # 预热加载，减少首次预测等待
    return info


# ============================================================================
# Gradio UI 构建
# ============================================================================
with gr.Blocks(title="电商评论情感分析") as demo:
    gr.Markdown("# 电商评论情感二分类 Demo")

    # ---- 模型选择行 ----
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=AVAILABLE_MODELS, value=BEST_MODEL,
            label="选择模型", interactive=True)
        model_info = gr.Markdown(
            f"Macro-F1: **{MODEL_STATS[BEST_MODEL]['macro_f1']:.4f}** | "
            f"Accuracy: **{MODEL_STATS[BEST_MODEL]['accuracy']:.4f}**")

    # ---- 输入区 ----
    inp = gr.Textbox(lines=4, placeholder="在此输入中文电商评论...", label="评论内容")
    btn = gr.Button("分析情感", variant="primary")

    # ---- 输出区 ----
    with gr.Row():
        lbl = gr.Label(label="情感概率", num_top_classes=2)
        result = gr.Markdown()

    # ---- 示例 ----
    gr.Examples(
        examples=[
            "产品质量很好，做工精细，物流也很快，非常满意！",
            "用了不到一个月就坏了，客服也不理人，太失望了。",
            "一般般吧，没想象中那么好用，凑合着用。",
        ],
        inputs=inp,
    )

    # ---- 事件绑定 ----
    # 点击按钮 → 执行预测
    btn.click(fn=predict, inputs=[inp, model_dropdown], outputs=[result, lbl])
    # 切换模型 → 更新指标信息 + 预热加载
    model_dropdown.change(fn=switch_model, inputs=model_dropdown, outputs=model_info)

if __name__ == "__main__":
    demo.launch()
