"""
数据采样与划分脚本（Step 1）
=============================================================================
功能：合并多个评论数据集 → 均衡正负样本 → 分层划分 8:1:1

处理流程：
  1. 加载并规范化各输入文件（支持 csv/tsv/json/jsonl）
  2. 自动检测数据格式（星级评分格式 vs 已有标签格式）
  3. 星级 → 二分类标签转换（star_to_label: 1-3→0, 4-5→1）
  4. 数据清洗（去空、去无效、只保留 0/1 标签）
  5. 合并所有数据集
  6. 下采样多数类，使正负样本 1:1 均衡
  7. 分层抽样划分 训练集:验证集:测试集 = 8:1:1
  8. 保存到 data/processed/

支持的输入格式：
  - 星级格式：包含"评论内容"/"review"等文本列 + "评分"/"star"等星级列
  - 标签格式：包含"text"/"comment"等文本列 + "label"标签列（0/1）

使用方法：
  python scripts/01_sampling.py                              # 使用默认文件
  python scripts/01_sampling.py --input a.csv b.csv          # 指定输入
  python scripts/01_sampling.py --no-balance                 # 跳过均衡步骤
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import SEED, PROCESSED_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, star_to_label

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 列名检测候选列表：按优先级顺序匹配，越靠前的候选名优先级越高
# ---------------------------------------------------------------------------
# 文本列候选名（支持中英文常见的列名）
TEXT_CANDIDATES = ["review", "评论内容", "comment", "text", "content", "评论标题"]
# 星级列候选名
STAR_CANDIDATES = ["评分", "star", "score", "rating", "stars", "星级"]
# 已有标签列候选名
LABEL_CANDIDATES = ["label"]


def detect_format(df: pd.DataFrame) -> dict:
    """自动检测 DataFrame 的数据格式

    按优先级依次尝试匹配候选列名，确定文本列和标签列的对应关系。
    检测策略：
      1. 先找文本列（在 TEXT_CANDIDATES 中按优先级匹配）
      2. 再找是否包含星级列 → 判定为 "star" 格式
      3. 否则找是否包含 label 列 → 判定为 "label" 格式

    Returns:
        {"text_col": 列名, "star_col"/"label_col": 列名, "format": "star"|"label"}
    """
    # --- 第一步：检测文本列 ---
    text_col = None
    for candidate in TEXT_CANDIDATES:
        for col in df.columns:
            # 同时支持精确匹配（candidate == col.lower()）和包含匹配（candidate in col）
            # 后者用于处理如"产品评论内容"这类列名
            if candidate == col.lower() or candidate in col:
                text_col = col
                break
        if text_col:
            break

    if text_col is None:
        raise RuntimeError(f"Cannot detect text column. Available: {list(df.columns)}")

    # --- 第二步：检测星级列 ---
    star_col = None
    for candidate in STAR_CANDIDATES:
        for col in df.columns:
            if col != text_col and (candidate == col.lower() or candidate in col):
                star_col = col
                break
        if star_col:
            break

    # --- 第三步：检测 label 列（仅在无星级列时尝试） ---
    label_col = None
    for candidate in LABEL_CANDIDATES:
        for col in df.columns:
            if col != text_col and candidate == col.lower():
                label_col = col
                break
        if label_col:
            break

    # --- 返回检测结果 ---
    if star_col:
        fmt = "star"
        result = {"text_col": text_col, "star_col": star_col, "format": fmt}
    elif label_col is not None:
        fmt = "label"
        result = {"text_col": text_col, "label_col": label_col, "format": fmt}
    else:
        raise RuntimeError(
            f"Cannot detect rating or label column. Available: {list(df.columns)}"
        )

    print(f"  {Path(df.attrs.get('path', '?')).name}: "
          f"fmt={fmt}, text='{text_col}'"
          f"{', rating=' + repr(star_col) if star_col else ''}"
          f"{', label=' + repr(label_col) if label_col else ''}")
    return result


def load_and_normalize(input_path: str) -> pd.DataFrame:
    """加载单个文件，检测格式，统一输出为 [text, label] 两列

    处理步骤：
      1. 根据文件后缀自动选择读取方式（csv/tsv/json/jsonl）
      2. 自动检测数据格式
      3. 星级格式 → 用 star_to_label() 转二分类
      4. 已有标签格式 → 转为数值
      5. 清洗：去空文本、去非 0/1 标签（如 neutral=2）

    Args:
        input_path: 数据文件路径

    Returns:
        仅含 "text" 和 "label" 两列的 DataFrame
    """
    path = Path(input_path)
    print(f"Loading: {path}")

    # 根据文件后缀选择加载方式
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # 用 attrs 记录原始路径，供 detect_format 打印日志用
    df.attrs["path"] = str(path)
    info = detect_format(df)
    text_col = info["text_col"]

    # 根据检测到的格式进行标签转换
    if info["format"] == "star":
        # 星级 → 二分类（star_to_label: 1-3→0, 4-5→1）
        df["label"] = df[info["star_col"]].apply(star_to_label)
    else:
        # 已有标签格式 → 转为数值
        df["label"] = pd.to_numeric(df[info["label_col"]], errors="coerce")

    # ---- 数据清洗 ----
    # 去除文本为空的行
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    # 去除标签为空的行
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    # 只保留 0/1 标签，过滤掉中间值（如某些数据集的 neutral=2）
    df = df[df["label"].isin([0, 1])]

    # 统一输出列
    df["text"] = df[text_col].astype(str).str.strip()
    return df[["text", "label"]].reset_index(drop=True)


def show_distribution(df: pd.DataFrame, title: str = ""):
    """以可视化文本条的方式打印标签分布

    输出示例：
      差评:   15,000 ( 50.0%) █████████████████████████
      好评:   15,000 ( 50.0%) █████████████████████████
      比例 (max/min): 1.0:1
    """
    label_names = {0: "差评", 1: "好评"}
    total = len(df)
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}: {total:,} rows")
        print(f"{'='*50}")
    else:
        print(f"\nTotal: {total:,} rows")

    label_dist = df["label"].value_counts().sort_index()
    for lbl, count in label_dist.items():
        pct = 100 * count / total
        # 用 █ 字符绘制简易条形图，每 2% 用 1 个 █
        bar = "█" * int(pct / 2)
        print(f"  {label_names[lbl]}: {count:>8,} ({pct:5.1f}%) {bar}")

    if len(label_dist) == 2:
        max_c = label_dist.max()
        min_c = label_dist.min()
        print(f"  比例 (max/min): {max_c / min_c:.1f}:1")


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """下采样多数类，使正负样本达到 1:1 均衡

    为什么要均衡？
      电商评论中好评（4-5星）通常占 80%+，差评（1-3星）占不到 20%，
      如果不均衡，模型会偏向预测好评，差评的召回率会很低。
      下采样多数类是最简单有效的均衡方法。

    注意：仅做下采样（减少多数类），而非上采样（复制少数类），
    因为上采样会增加过拟合风险。

    Args:
        df: 含 "text" 和 "label" 列的 DataFrame

    Returns:
        均衡后的 DataFrame（已随机打乱）
    """
    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]
    n_neg, n_pos = len(neg), len(pos)

    if n_neg == n_pos:
        print("\n  数据已均衡，无需下采样。")
        return df

    target = min(n_neg, n_pos)
    majority_name = "好评" if n_pos > n_neg else "差评"
    minority_name = "差评" if n_pos > n_neg else "好评"

    # 保留全部少数类，从多数类中随机下采样
    if n_neg < n_pos:
        keep = neg
        downsample = pos.sample(n=target, random_state=SEED)
    else:
        keep = pos
        downsample = neg.sample(n=target, random_state=SEED)

    print(f"\n  下采样 {majority_name}: {max(n_neg, n_pos):,} -> {target:,}")
    print(f"  保留全部 {minority_name}: {min(n_neg, n_pos):,}")
    print(f"  均衡后总量: {target * 2:,}")

    # 合并后随机打乱，避免正负样本聚集
    return pd.concat([keep, downsample], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    """分层抽样划分 8:1:1 并保存到 data/processed/

    划分策略：使用 sklearn 的 train_test_split 分两次完成
      第一次：df → train(80%) + temp(20%)
      第二次：temp → val(50% = 10%) + test(50% = 10%)

    使用 stratify 参数确保每个子集中正负样本比例一致。
    """
    # 分层抽样划分：保证划分后各集合的标签比例与原始一致
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=SEED)

    # 确保目标目录存在
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # 以 UTF-8 BOM 编码保存，确保 Excel 直接打开不乱码
    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    label_names = {0: "差评", 1: "好评"}
    print(f"\n{'='*50}")
    print(f"  划分结果 (8:1:1)")
    print(f"{'='*50}")
    for name, path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        sub = pd.read_csv(path)
        dist = sub["label"].value_counts().sort_index()
        parts = ", ".join(f"{label_names[k]}: {v:,}" for k, v in dist.items())
        print(f"  {name}: {len(sub):>8,} rows ({parts})")


def main():
    """主入口：解析参数 → 加载合并 → 均衡 → 划分保存"""
    # 默认输入文件：训练集 + 线上购物10分类数据集
    default_inputs = [
        "data/raw/训练集.csv",
        "data/raw/online_shopping_10_cats.csv",
    ]
    parser = argparse.ArgumentParser(
        description="Merge review datasets, balance pos/neg, stratified split 8:1:1"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", default=default_inputs,
        help="Input CSV files (default: 训练集.csv + online_shopping_10_cats.csv)"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip class balancing（保留原始分布不均衡状态）"
    )
    args = parser.parse_args()

    # --- 第 1 步：加载并规范化所有输入 ---
    dfs = []
    for p in args.input:
        dfs.append(load_and_normalize(p))

    # --- 第 2 步：合并所有数据集 ---
    merged = pd.concat(dfs, ignore_index=True)
    show_distribution(merged, "合并后（原始）")

    # --- 第 3 步：均衡正负样本 ---
    if not args.no_balance:
        merged = balance(merged)
        show_distribution(merged, "均衡后")

    # --- 第 4 步：分层划分并保存 ---
    split_and_save(merged)
    print("\n完成！")


if __name__ == "__main__":
    main()
