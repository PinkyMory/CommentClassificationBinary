# 电商评论二分类情感分析

基于多种方法（传统机器学习 / 深度学习 / 预训练模型）的**中文电商评论情感二分类**项目，将 1-3 星归为差评（negative），4-5 星归为好评（positive），并提供 Gradio 交互式 Web Demo。

## 项目结构

```
├── app/                        # Gradio Web Demo
│   ├── demo.py                 #   推理界面入口
│   └── model_loader.py         #   统一模型加载与预测接口
├── data/
│   ├── raw/                    # 原始数据（.csv/.tsv/.json/.jsonl）
│   ├── processed/              # 采样后的 train/val/test.csv
│   └── embeddings/             # 预训练词向量（可选）
├── checkpoints/                # 训练好的模型权重
├── outputs/
│   ├── results.csv             # 所有模型在测试集上的结果汇总
│   └── figures/                # 混淆矩阵、训练曲线、模型对比图
├── scripts/
│   ├── 01_sampling.py          # 数据采样：读原始数据 → 二值标注 → 分层划分
│   ├── 02_train_traditional.py # 传统 ML：TF-IDF + NB / SVM / XGBoost
│   ├── 03_train_dl.py          # 从头训练 DL：TextCNN + BiGRU-Attention
│   ├── 04_train_pretrained.py  # 微调预训练：BERT-base-Chinese + RoBERTa-wwm-ext
│   └── 05_evaluate_all.py      # 汇总所有模型结果，生成对比图表
├── src/
│   ├── config.py               # 全局配置（路径、超参、标签映射）
│   ├── preprocess.py           # 文本清洗 + jieba 分词
│   ├── dataset.py              # TokenizedDataset / 词表构建 / 词向量矩阵
│   ├── train_utils.py          # 训练循环、EarlyStopping、评估函数
│   ├── metrics.py              # 指标计算、结果写入 results.csv
│   ├── plot.py                 # 可视化（混淆矩阵、训练曲线、模型对比）
│   └── models/
│       ├── textcnn.py          # TextCNN：3/4/5-gram 多尺寸卷积核
│       └── bigru_attn.py       # BiGRU + Bahdanau 注意力机制
├── notebooks/                  # 探索性分析的 Jupyter Notebook（可选）
├── requirements.txt            # Python 依赖
├── setup.sh                    # 一键环境安装脚本
└── CLAUDE.md                   # 项目开发指南（给 AI 助手使用）
```

## 标签定义

| 评分 | 标签 | 含义 |
|------|------|------|
| 1-3 星 | 0 | 差评（negative） |
| 4-5 星 | 1 | 好评（positive） |

> 3 星评论归为差评（negative），不做排除处理。

## 数据均衡

`01_sampling.py` 默认合并 `训练集.csv` + `online_shopping_10_cats.csv`，通过下采样好评使正负比例达到 1:1。因此**默认训练时无需额外的类不平衡处理**。

如果使用其他未均衡的数据，可通过 `--balanced` 启用各脚本的类权重机制：

```bash
python scripts/02_train_traditional.py --balanced
python scripts/03_train_dl.py --balanced
python scripts/04_train_pretrained.py --balanced
```

各方法的 `--balanced` 对应策略：

| 方法 | --balanced 启用时 |
|------|-----------|
| LinearSVC | `class_weight='balanced'` |
| XGBoost | `compute_sample_weight('balanced')` |
| TextCNN / BiGRU | WeightedRandomSampler + 加权 CrossEntropyLoss |
| BERT / RoBERTa | 加权 CrossEntropyLoss |

**主要评价指标**：Macro-F1（在类别不平衡下比 Accuracy 更可靠）。

## 环境搭建

### 方式一：Conda 环境

```bash
conda create -n llm python=3.12
conda activate llm
pip install -r requirements.txt
```

### 方式二：一键脚本（AutoDL / Linux GPU 实例）

```bash
bash setup.sh
```

### 主要依赖

- **PyTorch** >= 2.11 + CUDA
- **Transformers** >= 5.9（HuggingFace）
- **scikit-learn** >= 1.8
- **XGBoost** >= 3.2
- **jieba** >= 0.42.1（中文分词）
- **Gradio** >= 6.15（Web Demo）
- **gensim** >= 4.4（词向量加载）

## 使用流程

### Step 1: 准备数据

将原始数据（CSV / TSV / JSON / JSONL）放入 `data/raw/`，确保包含评论内容列和评分列。

### Step 2: 数据采样

```bash
python scripts/01_sampling.py --input data/raw/训练集.csv
```

脚本会自动检测评论列和评分列，也可以手动指定：

```bash
python scripts/01_sampling.py --input data/raw/训练集.csv --text-col 评论内容 --star-col 评分
```

输出：`data/processed/train.csv`、`val.csv`、`test.csv`（分层 8:1:1 划分）。

### Step 3: 训练模型

**传统机器学习（TF-IDF + NB / SVM / XGBoost）**：

```bash
python scripts/02_train_traditional.py
```

**深度学习从头训练（TextCNN + BiGRU-Attention）**：

```bash
python scripts/03_train_dl.py --model all          # 训练全部
python scripts/03_train_dl.py --model textcnn      # 只训练 TextCNN
python scripts/03_train_dl.py --model bigru_attn   # 只训练 BiGRU
```

可选：指定预训练词向量路径以提升效果：

```bash
python scripts/03_train_dl.py --wv-path data/embeddings/sgns.sogou.word
```

**微调预训练模型（BERT + RoBERTa）**：

```bash
python scripts/04_train_pretrained.py --model all      # 训练全部
python scripts/04_train_pretrained.py --model bert     # 只训练 BERT
python scripts/04_train_pretrained.py --model roberta  # 只训练 RoBERTa
```

### Step 4: 汇总结果

```bash
python scripts/05_evaluate_all.py
```

读取 `outputs/results.csv`，按 Macro-F1 排序打印所有模型对比，并生成：
- `outputs/figures/model_comparison.png` — 各模型 Accuracy / Macro-F1 / Weighted-F1 对比
- `outputs/figures/per_class_f1.png` — 各模型在差评/好评上的 F1 对比

### Step 5: 启动 Web Demo

```bash
python app/demo.py
```

启动 Gradio 界面，自动加载 results.csv 中 Macro-F1 最高的模型进行推理。

## 模型概览

| 类别 | 模型 | 关键技术 | 说明 |
|------|------|---------|------|
| 传统 ML | MultinomialNB | TF-IDF + 朴素贝叶斯 | 简单高效，基准模型 |
| 传统 ML | LinearSVC | TF-IDF + 线性 SVM | 带概率校准（CalibratedClassifierCV） |
| 传统 ML | XGBoost | TF-IDF + 梯度提升树 | 带样本权重处理不平衡 |
| DL 从头训练 | TextCNN | 多尺寸卷积核 (3/4/5-gram) | 可加载预训练词向量 |
| DL 从头训练 | BiGRU-Attention | 双向 GRU + Bahdanau 注意力 | 捕捉长距离语义依赖 |
| 预训练 | BERT-base-Chinese | 12 层 Transformer 编码器 | 中文原生 BERT |
| 预训练 | RoBERTa-wwm-ext | 全词掩码 + 更大训练语料 | BERT 改进版，通常效果更优 |

## 配置说明

所有超参数集中在 [src/config.py](src/config.py)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEED` | 42 | 全局随机种子 |
| `USE_ALL_DATA` | True | 是否使用全部数据 |
| `TRAIN_RATIO / VAL_RATIO / TEST_RATIO` | 0.8 / 0.1 / 0.1 | 数据集划分比例 |
| `MAX_SEQ_LEN` | 128 | TextCNN/BiGRU 最大序列长度 |
| `MAX_SEQ_LEN_BERT` | 256 | BERT/RoBERTa 最大序列长度 |
| `TFIDF_MAX_FEATURES` | 5000 | TF-IDF 特征维度 |
| `EMBEDDING_DIM` | 300 | 词向量维度 |
| `BATCH_SIZE_DL` | 64 | DL 从头训练批次大小 |
| `EPOCHS_DL` | 30 | DL 从头训练最大轮数 |
| `BATCH_SIZE_PRETRAINED` | 16 | 预训练模型批次大小 |
| `EPOCHS_PRETRAINED` | 5 | 预训练模型微调轮数 |
| `LR_PRETRAINED` | 2e-5 | 预训练模型学习率 |

## 技术要点

- **预分词**：`TokenizedDataset` 在 `__init__` 中完成 jieba 分词，避免每个 epoch 重复分词。
- **模型缓存**：预训练模型首次下载后使用 `local_files_only=True` 优先加载本地缓存，避免超时。
- **无头服务器兼容**：`matplotlib` 使用 `Agg` 后端，可在无 GUI 的 GPU 服务器上正常保存图表。
- **脚本独立性**：每个训练脚本是独立入口点，共享 `src/` 中的评估和可视化模块，但各自拥有独立的模型定义和训练逻辑。

## 扩展方向

- 新增预训练模型：在 [04_train_pretrained.py](scripts/04_train_pretrained.py) 的 `main()` 中添加新分支，在 `config.py` 中添加模型名称常量。
- 新增 DL 模型：在 `src/models/` 中创建新文件，在 [03_train_dl.py](scripts/03_train_dl.py) 中注册即可。
- 多分类扩展：修改 `star_to_label()` 为三分类，调整各脚本的 `num_classes` 参数。
