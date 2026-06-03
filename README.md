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

## 模型详情

### 1. MultinomialNB（多项式朴素贝叶斯）

**原理**：基于贝叶斯定理，假设特征（TF-IDF 词频）服从多项式分布。对于一篇评论，计算它在正/负类下的后验概率，取概率最大的类别。

**实现细节**：
- 输入：TF-IDF 加权词频向量（5,000 维）
- 平滑系数 `alpha=0.5`，防止零概率问题
- 不做额外加权，依赖朴素条件独立假设

**优点**：训练极快（秒级），对短文本效果尚可，可解释性强。**缺点**：特征独立假设过强，忽视词序和上下文，长文本容易失准。

---

### 2. LinearSVC（线性支持向量机）

**原理**：在高维 TF-IDF 空间中寻找一个最大间隔超平面，将正负样本分开。优化目标是 hinge loss + L2 正则化。

**实现细节**：
- 输入：TF-IDF 加权词频向量（5,000 维）
- 超参：`C=1.0`，`max_iter=2000`，`dual=False`（样本数 > 特征数时使用 primal 形式）
- 外层包裹 `CalibratedClassifierCV` 做 Platt 概率校准，输出可解释的置信度

**优点**：高维稀疏数据上表现稳定，收敛性好。**缺点**：线性决策边界，无法捕捉非线性语义关系。

---

### 3. XGBoost（极端梯度提升）

**原理**：基于梯度提升决策树（GBDT）的集成方法，每棵新树拟合前一棵树的残差。目标函数 = 损失函数 + 正则项（叶子节点数 + L2 权重）。

**实现细节**：
- 输入：TF-IDF 词频 5,000 维 + 字符级 n-gram 5,000 维 = **10,000 维**（`--tune-xgboost` 模式）
- 超参（默认）：`n_estimators=200`，`max_depth=6`，`learning_rate=0.1`
- 可选 `--tune-xgboost`：启用 `RandomizedSearchCV`（50 组 × 3 折，macro-F1 评分），搜索 `n_estimators`、`max_depth`、`subsample`、正则化系数等
- 字符级特征：按字切分 n-gram (1,2,3)，捕获字面模式（如 "不好用"、"太差了"）

**优点**：可处理非线性关系，内置缺失值处理，对表格型特征效果好。**缺点**：对高维稀疏文本特征不如线性模型自然，需要特征工程辅助。

---

### 4. TextCNN

**原理**：将文本视为一维图像，用不同尺寸的卷积核（3/4/5-gram）在词向量上滑动，提取局部 n-gram 语义特征。多个卷积核的输出拼接后送入全连接层分类。

**架构**：
```
Embedding (vocab_size × 300) → Conv2D(1, 100, kernel=(3/4/5, 300)) → ReLU → MaxPool1D
                              → 拼接 3 个池化结果 → Dropout(0.5) → Linear(300 → 2)
```

**实现细节**：
- 词向量：加载预训练 SGNS 300 维中文词向量，覆盖约 70-80% 词汇，微调模式（`freeze_embeddings=False`）
- 卷积核数：每种尺寸 100 个，共 300 个特征
- 训练：Adam 优化器，`lr=1e-3`，`ReduceLROnPlateau` 调度，EarlyStopping（patience=5）

**优点**：结构简单、训练速度快，对局部短语模式敏感。**缺点**：卷积核视野有限（最多 5-gram），无法捕捉长距离依赖。

---

### 5. BiGRU-Attention（双向门控循环网络 + 注意力）

**原理**：BiGRU 双向编码序列上下文，Bahdanau 加性注意力机制计算每个位置的权重，加权求和得到句子的分布式表示。

**架构**：
```
Embedding (vocab_size × 300) → BiGRU(300 → 128×2, num_layers=1)
                              → Attention(hidden_dim=256) → 加权上下文向量
                              → Dropout(0.5) → Linear(256 → 2)
```

**注意力机制**：对 BiGRU 输出的每个时间步，通过一个前馈网络 `v^T tanh(W·h_t)` 计算注意力分数，Softmax 归一化后加权求和。注意力权重反映了模型"关注"句子的哪些部分。

**实现细节**：
- 隐藏层大小：128（双向总计 256）
- 训练配置与 TextCNN 相同

**优点**：可捕捉长距离上下文，注意力权重可解释（可视化模型关注的关键词）。**缺点**：序列模型无法并行，训练比 TextCNN 慢。

---

### 6. BERT-base-Chinese

**原理**：12 层 Transformer 编码器，通过 MLM（掩码语言模型）+ NSP（下一句预测）在大规模中文语料上预训练。微调时在 [CLS] 表征上接一个线性分类头。

**实现细节**：
- 参数量：~110M
- 词汇表：21,128 个中文字符级 token
- 最大长度：256 tokens
- 训练：AdamW，`lr=2e-5`，warmup 占比 10%，weight decay 0.01
- EarlyStopping（patience=2），best model 按 macro-F1 保存
- 批次大小：16，epochs：5
- 使用混合精度训练（FP16）

**优点**：语义理解能力强，开箱即用效果好。**缺点**：推理慢，显存占用高，不适合实时场景。

---

### 7. RoBERTa-wwm-ext

**原理**：BERT 架构的改进版，取消了 NSP 任务，使用更大的 batch size 和训练数据。**WWM（Whole Word Masking）**：对中文整词进行掩码而非单字，迫使模型学习更完整的语义信息。**Ext**：在 CLUECorpusSmall + 百科 + 问答等更大语料上训练。

**实现细节**：
- 模型路径：`hfl/chinese-roberta-wwm-ext`（哈工大/讯飞联合出品）
- 参数量和训练超参与 BERT 完全相同
- 唯一区别：预训练策略更优，通常比原生 BERT 高出 0.5-1 个 F1 点

**优点**：中文场景下效果最优，句子级语义理解更精准。**缺点**：与 BERT 相同的推理开销。

---

### 实验结果对比

| 模型 | Macro-F1 | Accuracy | 差评 F1 | 好评 F1 | 训练时间 |
|------|----------|----------|---------|---------|----------|
| RoBERTa-wwm-ext | **0.9075** | 0.9075 | 0.9075 | 0.9076 | ~20 min |
| BERT-base-Chinese | 0.9026 | 0.9026 | 0.9027 | 0.9025 | ~18 min |
| BiGRU-Attention | 0.8903 | 0.8903 | 0.8895 | 0.8911 | ~8 min |
| TextCNN | 0.8808 | 0.8808 | 0.8827 | 0.8788 | ~5 min |
| LinearSVC | 0.8477 | 0.8477 | 0.8453 | 0.8501 | ~3 min |
| MultinomialNB | 0.8387 | 0.8387 | 0.8346 | 0.8427 | < 1 min |
| XGBoost | 0.8114 | 0.8116 | 0.8173 | 0.8056 | ~2 min |

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
