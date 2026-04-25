#!/usr/bin/env python3
"""
生成 CRN 所需的训练和测试文件列表（包含正确相对路径）。
自动检测项目根目录。
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

CRN_DATA_DIR = os.path.join(PROJECT_ROOT, "baselines", "CRN-causal", "data", "datasets")
TR_DIR_ABS = os.path.join(CRN_DATA_DIR, "tr")
TT_DIR_ABS = os.path.join(CRN_DATA_DIR, "tt")

FILELIST_DIR = os.path.join(PROJECT_ROOT, "baselines", "CRN-causal", "filelists")

# 列表文件中的路径应相对于 scripts 目录（CRN训练时所处目录）
TR_DIR_REL = os.path.join("..", "data", "datasets", "tr")
TT_DIR_REL = os.path.join("..", "data", "datasets", "tt")


def main():
    os.makedirs(FILELIST_DIR, exist_ok=True)

    if os.path.exists(TR_DIR_ABS):
        tr_files = sorted([f for f in os.listdir(TR_DIR_ABS) if f.endswith(".ex")])
        tr_list_path = os.path.join(FILELIST_DIR, "tr_list.txt")
        with open(tr_list_path, "w") as f:
            for name in tr_files:
                rel_path = os.path.join(TR_DIR_REL, name)
                f.write(f"{rel_path}\n")
        print(f"训练集列表生成: {tr_list_path} (共 {len(tr_files)} 个文件)")
    else:
        print(f"警告：训练集目录 {TR_DIR_ABS} 不存在，跳过 tr_list.txt")

    if os.path.exists(TT_DIR_ABS):
        tt_files = sorted([f for f in os.listdir(TT_DIR_ABS) if f.endswith(".ex")])
        tt_list_path = os.path.join(FILELIST_DIR, "tt_list.txt")
        with open(tt_list_path, "w") as f:
            for name in tt_files:
                rel_path = os.path.join(TT_DIR_REL, name)
                f.write(f"{rel_path}\n")
        print(f"测试集列表生成: {tt_list_path} (共 {len(tt_files)} 个文件)")
    else:
        print(f"警告：测试集目录 {TT_DIR_ABS} 不存在，跳过 tt_list.txt")


if __name__ == "__main__":
    main()
