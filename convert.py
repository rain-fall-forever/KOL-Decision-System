# -*- coding: utf-8 -*-
"""
SNAP ego-Facebook 原始边表 -> 本系统可直接读取的 CSV
========================================================
原始文件每行是 "节点A 节点B"（空格分隔、无表头），本脚本补上表头并清洗，
输出列固定为 source,target。

用法（任意工作目录下都能跑，路径按脚本所在位置解析）：
    python convert.py
    python convert.py --input data/facebook_combined.txt --output data/real_facebook_data.csv

默认：
    输入  data/facebook_combined.txt
    输出  data/real_facebook_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# 以脚本所在目录为基准，避免"在别的目录下执行就找不到文件"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "facebook_combined.txt"
DEFAULT_OUTPUT = BASE_DIR / "data" / "real_facebook_data.csv"


def convert(src: Path, dst: Path) -> pd.DataFrame:
    """读取空格分隔的原始边表，清洗后写出 CSV，返回清洗结果。"""
    if not src.exists():
        raise FileNotFoundError(
            f"找不到原始数据：{src}\n"
            f"请确认 facebook_combined.txt 已放在 data/ 目录下，"
            f"或使用 --input 指定路径。"
        )

    # sep=r"\s+" 兼容单空格 / 多空格 / 制表符；dtype=str 保留原始 ID
    df = pd.read_csv(src, sep=r"\s+", header=None, names=["source", "target"],
                     dtype=str, comment="#", engine="python")
    n_raw = len(df)

    # 清洗：空值 / 自环 / 重复边
    df = df.dropna(subset=["source", "target"])
    df = df[df["source"] != df["target"]]
    df = df.drop_duplicates().reset_index(drop=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8")

    print(f"原始边数：{n_raw:,}")
    print(f"清洗后  ：{len(df):,} 条边，{len(set(df['source']) | set(df['target'])):,} 个节点")
    print(f"已保存  ：{dst}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="SNAP ego-Facebook 边表转 CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="原始 txt 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 csv 路径")
    args = parser.parse_args()

    try:
        convert(args.input, args.output)
    except Exception as exc:
        print(f"转换失败：{type(exc).__name__} - {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
