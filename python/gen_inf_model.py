#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import torch
from model import EnhancedResNetPolicyValueNetwork

# ====== 設定 ======
STATE_DICT_DIR   = r"G:\マイドライブ\sgf\models"       # state_dict ファイル置き場
MODEL_OUTPUT_DIR = r"D:\igo\simple2_sdl\x64\Release"   # 推論モデル保存先
IN_CHANNELS      = 17
NUM_CHANNELS     = 256
BOARD_SIZE       = 19
NUM_BLOCKS       = 40       # ResNet の residual block 数を固定
PREFIX           = "4-40"   # state_dict ファイル名の接頭辞
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================

def find_latest_state_dict(directory: str) -> str:
    """
    指定ディレクトリ内から model_{PREFIX}_acc_*.pt のうち
    最終更新日時が最新のものを返します。
    """
    # model_{PREFIX}_acc_*.pt にマッチするファイルを検索
    pattern = f"model_{PREFIX}_acc_*.pt"
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def save_inference_model(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str
) -> None:
    """
    PyTorch モデルを TorchScript にトレースして保存します。
    1) model を評価モードに設定
    2) ダミー入力でトレース
    3) CPU に移して保存
    4) モデルを元のデバイスに戻す
    """
    model.eval()
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = device
    print(f"[INFO] Original model device: {orig_device}")
    
    model.to(device)
    dummy = torch.randn(1, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
    traced = torch.jit.trace(model, dummy)
    traced_cpu = traced.to("cpu")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_OUTPUT_DIR, model_name)
    torch.jit.save(traced_cpu, save_path)
    print(f"[INFO] Inference model saved: {save_path}")
    model.to(orig_device)


if __name__ == "__main__":
    # 1) 最新の state_dict を探す
    state_dict_path = find_latest_state_dict(STATE_DICT_DIR)
    print(f"[INFO] Found state_dict: {state_dict_path}")

    # 2) ファイル名から数字部分を抽出してサフィックスを生成
    basename = os.path.basename(state_dict_path)            # model_4-30_acc_0.09254.pt
    stem     = os.path.splitext(basename)[0]                # model_4-30_acc_0.09254
    nums     = re.findall(r"\d+\.\d+|\d+", stem)        # ['4','30','0.09254']
    suffix   = "_".join(nums)                             # '4_30_0.09254'
    inference_name = f"inference_model_{suffix}.pt"

    # 3) モデル生成＆state_dict 読み込み
    model = EnhancedResNetPolicyValueNetwork(
        board_size=BOARD_SIZE,
        in_channels=IN_CHANNELS,
        num_channels=NUM_CHANNELS,
        num_blocks=NUM_BLOCKS
    )
    state_dict = torch.load(state_dict_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded state_dict from {state_dict_path}")

    # 4) 推論モデル生成・保存
    save_inference_model(model, DEVICE, inference_name)
