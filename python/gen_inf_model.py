#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import torch
from model import EnhancedResNetPolicyValueNetwork

# ====== 設定 ======
STATE_DICT_DIR = r"G:\マイドライブ\sgf\models"            # state_dict ファイル置き場
MODEL_OUTPUT_DIR = r"D:\igo\simple2_sdl\x64\Release"        # 推論モデル保存先
NUM_CHANNELS = 17
BOARD_SIZE   = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_state_dict(directory: str, pattern: str = "model_4_*.pt") -> str:
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def save_inference_model(model: torch.nn.Module,
                         device: torch.device,
                         model_name: str) -> None:
    model.eval()
    # 元のデバイスを保持
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = device

    # ① モデルをデバイスに移動
    model.to(device)
    # ② ダミー入力
    dummy = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
    # ③ トレース
    traced = torch.jit.trace(model, dummy)
    # ④ CPU に移して保存
    traced_cpu = traced.to("cpu")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_OUTPUT_DIR, model_name)
    torch.jit.save(traced_cpu, save_path)
    print(f"[INFO] Inference model saved: {save_path}")
    # ⑤ 元のデバイスに戻す
    model.to(orig_device)


if __name__ == "__main__":
    # 1) 最新の state_dict を探す
    state_dict_path = find_latest_state_dict(STATE_DICT_DIR)
    print(f"[INFO] Found state_dict: {state_dict_path}")

    # 2) 数字部分を抽出（例: ['4','0.09254'] → '4_0.09254'）
    basename = os.path.basename(state_dict_path)            # model_4_acc_0.09254.pt
    stem     = os.path.splitext(basename)[0]                # model_4_acc_0.09254
    nums     = re.findall(r"\d+\.\d+|\d+", stem)            # ['4','0.09254']
    suffix   = "_".join(nums)                               # '4_0.09254'
    inference_name = f"inference_model_{suffix}.pt"         # inference_model_4_0.09254.pt

    # 3) モデル生成＆state_dict読み込み
    model = EnhancedResNetPolicyValueNetwork(
        board_size=BOARD_SIZE,
        in_channels=NUM_CHANNELS,
        num_channels=256,
        num_blocks=40
    )
    state_dict = torch.load(state_dict_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded state_dict from {state_dict_path}")

    # 4) 推論モデル生成・保存
    save_inference_model(model, DEVICE, inference_name)
