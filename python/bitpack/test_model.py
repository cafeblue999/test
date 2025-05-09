import os
import pytest
import torch
from torch.utils.data import DataLoader
os.environ['USE_TPU'] = '0'
os.environ['MODEL_OUTPUT_DIR'] = './tests_tmp_models'
# または、config モジュールの変数を書き換え
import config
config.MODEL_OUTPUT_DIR = os.environ['MODEL_OUTPUT_DIR']
from pathlib import Path
Path(config.MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# あなたのデータセットクラスをインポート
from dataset import AlphaZeroSGFDatasetPreloaded

# --- テスト用ダミーサンプルを定義（実際はSGFから生成済みのsamplesを読み込む） ---
# samples は [(board_tensor, policy_index, value), …] の形と仮定
dummy_board = torch.zeros(19, 19, dtype=torch.int8)  # 全空き盤面
dummy_policy_idx = 3  # 適当な整数
dummy_value = 0.5
samples = [(dummy_board, dummy_policy_idx, dummy_value) for _ in range(10)]

# データセットのインスタンス化
dataset = AlphaZeroSGFDatasetPreloaded(samples)

def test_policy_target_type_and_shape():
    board, policy_target, value_target = dataset[0]

    # 1) board の形状チェック
    assert isinstance(board, torch.Tensor)
    assert board.shape == (19, 19)

    # 2) policy_target が LongTensor かつスカラー or (361,) か
    assert isinstance(policy_target, torch.Tensor)
    assert policy_target.dtype in (torch.long, torch.int64)
    # インデックス表現ならスカラー、ワンホットならベクトル
    assert policy_target.ndim in (0, 1)
    if policy_target.ndim == 0:
        idx = int(policy_target)
        assert 0 <= idx < 19*19
    else:
        # ワンホットなら合計が1
        assert policy_target.numel() == 19*19
        assert policy_target.sum().item() == 1

def test_policy_label_on_empty_point():
    board, policy_target, _ = dataset[0]
    # インデックス表現と仮定
    idx = int(policy_target)
    y, x = divmod(idx, 19)
    # board[y, x] が空き(0)であること
    assert board[y, x] == 0

def test_dataloader_batch_shapes():
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    boards, policy_targets, value_targets = batch

    # boards は (B, 19, 19)
    assert boards.shape == (4, 19, 19)
    # policy_targets は (B,) または (B,361)
    assert policy_targets.ndim in (1, 2) and policy_targets.size(0) == 4
    # value_targets は (B,)
    assert value_targets.shape == (4,)

if __name__ == "__main__":
    pytest.main()
