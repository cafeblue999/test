#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
(詳細な説明は省略されていますが、全体としては囲碁棋譜（SGF）のパース、前処理、
ディープラーニングモデルの定義、学習ループ、チェックポイント保存・復元、
およびTPUやGoogle Colabを利用する際の環境設定を行うコード)
"""

# ===== 固定定義：環境切り替え用フラグ =====
# TPU、Colabで実行するかどうかのフラグを定義
USE_TPU = True
USE_COLAB = True

# ------------------------------
# 必要なライブラリのインポート
# ------------------------------
import os, re, pickle, zipfile, random, numpy as np, configparser, argparse, functools
from tqdm import tqdm  # 進捗表示用ライブラリ
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist  # 分散学習用

# TPU利用時に必要なtorch_xla関連モジュールをインポート
if USE_TPU:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend

# Google Colab利用時にGoogle Driveのマウントを試みる
if USE_COLAB:
    try:
        # 既にDriveがマウントされている場合の解除
        os.system("fusermount -u /content/drive")
    except Exception as e:
        print("Google Drive unmount failed:", e)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        print("Google Colab module not found.")

# プログレスバーのフォーマット設定
bar_fmt = "{l_bar}{bar}| {n:>6d}/{total:>6d} [{elapsed}<{remaining}, {rate_fmt}]"

# ==============================
# デバイス設定
# ==============================
if USE_TPU:
    # TPUデバイスの取得
    device = xm.xla_device()
    # 分散処理が初期化されていなければ初期化
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method='xla://')
else:
    # GPUが利用可能であればCUDA、なければCPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# ディレクトリ設定
# ==============================
if USE_COLAB:
    # Colab環境用のディレクトリ設定
    BASE_DIR = "/content/drive/My Drive/sgf"
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf_KK")
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint2.pt")
else:
    # ローカル環境用のディレクトリ設定（例：Windows環境）
    BASE_DIR = r"D:\igo\simple2"
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf")
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint2.pt")

# モデル出力用ディレクトリが存在しなければ作成
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# ==============================
# DummyLogger クラス（ログ出力用）
# ==============================
from datetime import datetime, timedelta, timezone
# 日本標準時(JST)のタイムゾーン設定
JST = timezone(timedelta(hours=9), 'JST')
class DummyLogger:
    # infoレベルのログ出力：タイムスタンプ付き
    def info(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} INFO: {message}", *args, **kwargs)
    # warningレベルのログ出力
    def warning(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} WARNING: {message}", *args, **kwargs)
    # errorレベルのログ出力
    def error(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} ERROR: {message}", *args, **kwargs)

# ロガーインスタンスの生成
sgf_logger = DummyLogger()
train_logger = DummyLogger()

# ==============================
# 設定ファイル読み込み関数
# ==============================
def load_config(config_path):
    # 設定ファイルをパースするためのConfigParserの生成
    config = configparser.ConfigParser()
    config.read(config_path)
    try:
        # 各セクションから必要なパラメータを取得（存在しない場合はfallback値を利用）
        BOARD_SIZE = int(config.get("BOARD", "board_size", fallback="19"))
        HISTORY_LENGTH = int(config.get("DATA", "history_length", fallback="8"))

        # チャンネル数は履歴の2倍+現在のプレイヤー情報
        NUM_CHANNELS = 2 * HISTORY_LENGTH + 1

        # 全手数は盤上のマス＋パス（最後の1手）
        NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
        num_residual_blocks = int(config.get("MODEL", "num_residual_blocks", fallback="20"))
        model_channels = int(config.get("MODEL", "model_channels", fallback="256"))
        num_epochs = int(config.get("TRAIN", "num_epochs", fallback="100"))
        batch_size = int(config.get("TRAIN", "batch_size", fallback="256"))
        learning_rate = float(config.get("TRAIN", "learning_rate", fallback="0.001"))
        patience = int(config.get("TRAIN", "patience", fallback="10"))
        factor = float(config.get("TRAIN", "factor", fallback="0.8"))
        number_max_files = int(config.get("TRAIN", "number_max_files", fallback="256"))
    except Exception as e:
        train_logger.error(f"Error reading configuration: {e}")
        exit(1)

    # 辞書形式でパラメータを返す
    return {
        "BOARD_SIZE": BOARD_SIZE,
        "HISTORY_LENGTH": HISTORY_LENGTH,
        "NUM_CHANNELS": NUM_CHANNELS,
        "NUM_ACTIONS": NUM_ACTIONS,
        "num_residual_blocks": num_residual_blocks,
        "model_channels": model_channels,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "factor": factor,
        "number_max_files": number_max_files
    }

# 設定ファイルのパス（BASE_DIR以下にあると仮定）
CONFIG_PATH = os.path.join(BASE_DIR, "config_py.ini")
# 設定をロードし、パラメータを辞書に格納
config_params = load_config(CONFIG_PATH)

# ロードしたパラメータをグローバル変数に代入
BOARD_SIZE = config_params["BOARD_SIZE"]
HISTORY_LENGTH = config_params["HISTORY_LENGTH"]
NUM_CHANNELS = config_params["NUM_CHANNELS"]
NUM_ACTIONS = config_params["NUM_ACTIONS"]
num_residual_blocks = config_params["num_residual_blocks"]
model_channels = config_params["model_channels"]
num_epochs = config_params["num_epochs"]
batch_size = config_params["batch_size"]
learning_rate = config_params["learning_rate"]
patience = config_params["patience"]
factor = config_params["factor"]
number_max_files = config_params["number_max_files"]

# ログ出力：読み込んだ設定の確認
train_logger.info("==== Loaded Configuration ====")
train_logger.info(f"Config file: {CONFIG_PATH}")
train_logger.info(f"BOARD_SIZE: {BOARD_SIZE}")
train_logger.info(f"HISTORY_LENGTH: {HISTORY_LENGTH}")
train_logger.info(f"NUM_CHANNELS: {NUM_CHANNELS}")
train_logger.info(f"NUM_ACTIONS: {NUM_ACTIONS}")
train_logger.info(f"num_residual_blocks: {num_residual_blocks}")
train_logger.info(f"model_channels: {model_channels}")
train_logger.info(f"num_epochs: {num_epochs}")
train_logger.info(f"batch_size: {batch_size}")
train_logger.info(f"learning_rate: {learning_rate}")
train_logger.info(f"patience: {patience}")
train_logger.info(f"factor: {factor}")
train_logger.info(f"number_max_files: {number_max_files}")
train_logger.info("===============================")

# ==============================
# ネットワーク定義
# ==============================
# ResidualBlock : 標準的な残差ブロック。入力と出力のテンソルのサイズは同じで、
# 畳み込み→バッチ正規化→ReLUを2回適用し、最終的に入力と出力を足し合わせる。
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 畳み込み層1
        self.bn1 = nn.BatchNorm2d(channels)  # バッチ正規化1
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 畳み込み層2
        self.bn2 = nn.BatchNorm2d(channels)  # バッチ正規化2
    def forward(self, x):
        residual = x  # 入力を保持（スキップ接続）
        out = F.relu(self.bn1(self.conv1(x)))  # 畳み込み→正規化→ReLU
        out = self.bn2(self.conv2(out))  # 再度畳み込み→正規化
        out += residual  # 入力と加算（残差接続）
        return F.relu(out)  # 最後にReLU適用して出力

# DilatedResidualBlock : 拡張畳み込みを用いた残差ブロック。
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        super(DilatedResidualBlock, self).__init__()
        # 拡張畳み込み（dilation指定）を適用
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        # 通常の畳み込み層
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# SelfAttention : セルフアテンション機構を実現する層。
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # クエリ、キー、バリューの線形変換（1x1畳み込み）を定義
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 出力のスケールパラメータ
        self.softmax = nn.Softmax(dim=-1)  # ソフトマックス関数
    def forward(self, x):
        batch, C, H, W = x.size()
        # クエリを計算し形状変換（バッチ×(H×W)×(C//8)）
        proj_query = self.query_conv(x).view(batch, -1, H * W).permute(0, 2, 1)
        # キーを計算し形状変換（バッチ×(C//8)×(H×W)）
        proj_key = self.key_conv(x).view(batch, -1, H * W)
        # バッチ行列積によりエネルギーを計算
        energy = torch.bmm(proj_query, proj_key)
        # 各位置ごとに正規化（アテンションマップ）
        attention = self.softmax(energy)
        # バリューを計算して形状変換
        proj_value = self.value_conv(x).view(batch, -1, H * W)
        # バッチ行列積により出力を計算（アテンションの転置を適用）
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        # スケールパラメータを乗じ、入力を足して出力
        return self.gamma * out + x

# EnhancedResNetPolicyValueNetwork : 改良版残差ネットワークを基に
# ポリシーヘッドとバリューヘッドを組み合わせたネットワーク
class EnhancedResNetPolicyValueNetwork(nn.Module):
    def __init__(self, board_size, num_channels, num_residual_blocks, in_channels):
        super(EnhancedResNetPolicyValueNetwork, self).__init__()
        self.board_size = board_size
        # 入力処理：3x3畳み込みと正規化
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        # 複数の残差ブロック（定期的に拡張畳み込みブロックを挿入）
        blocks = []
        for i in range(num_residual_blocks):
            if i % 4 == 0:
                blocks.append(DilatedResidualBlock(num_channels, dilation=2))
            else:
                blocks.append(ResidualBlock(num_channels))
        self.residual_blocks = nn.Sequential(*blocks)
        # セルフアテンション層
        self.attention = SelfAttention(num_channels)
        # ポリシーヘッド
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.dropout_policy = nn.Dropout(p=0.5)
        # 全結合層により盤面上の各マス＋パスに対応する出力（行動数）を得る
        self.fc_policy = nn.Linear(2 * board_size * board_size, NUM_ACTIONS)
        # バリューヘッド
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        # 隠れ層の全結合
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.dropout_value = nn.Dropout(p=0.5)
        # 出力層：2ユニット、1はvalue、もう1つはmargin
        self.fc_value2 = nn.Linear(64, 2)
    def forward(self, x):
        # 入力を処理（畳み込み→正規化→ReLU）
        x = F.relu(self.bn_input(self.conv_input(x)))
        # 残差ブロックの適用
        x = self.residual_blocks(x)
        # セルフアテンションによる特徴変換
        x = self.attention(x)
        # ---- ポリシーヘッドの処理 ----
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = self.dropout_policy(p)
        p = p.view(p.size(0), -1)  # フラット化して全結合層へ
        p = self.fc_policy(p)
        p = F.log_softmax(p, dim=1)  # 対数ソフトマックスで正規化
        # ---- バリューヘッドの処理 ----
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = self.dropout_value(v)
        out = self.fc_value2(v)
        # 1ユニット目を[tanh]により[-1, 1]に収め、残りをmarginとして出力
        value = torch.tanh(out[:, 0])
        margin = out[:, 1]
        return p, (value, margin)

# ==============================
# SGFパーサー＆前処理関数
# ==============================
def parse_sgf(sgf_text):
    """
    SGF形式の文字列をパースして、棋譜のノード情報を抽出する。
    ・先頭と末尾の括弧を除去
    ・セミコロンでノードに分割し、各ノードのプロパティを正規表現で抽出
    """
    sgf_text = sgf_text.strip()  # 前後の空白除去
    if sgf_text.startswith('(') and sgf_text.endswith(')'):
        sgf_text = sgf_text[1:-1]  # 外側の括弧を除去

    # セミコロンで分割し、空でない部分のみ抽出
    parts = [part for part in sgf_text.split(';') if part.strip()]
    nodes = []

    # プロパティのキーと値をマッチする正規表現パターン
    prop_pattern = re.compile(r'([A-Z]+)
([^
]*)\]')
    for part in parts:
        props = {}
        # 正規表現でプロパティを抽出し、キーはUTF-8エンコード、値はリストに格納
        for m in prop_pattern.finditer(part):
            key = m.group(1).encode('utf-8')
            value = m.group(2)
            props[key] = [value.encode('utf-8')]
        nodes.append(props)
    if not nodes:
        raise ValueError("No nodes found in SGF file")

    # 最初のノードをroot、残りをnodesとして返す
    return {"root": nodes[0], "nodes": nodes[1:]}

def build_input_from_history(history, current_player, board_size, history_length):
    """
    棋譜の履歴からネットワークの入力テンソルを構築する。
    ・各履歴について、黒石か白石かをチャネルとして展開（各盤面は2チャネル）
    ・さらに現在のプレイヤー情報を1チャネル追加
    ・最終的に [チャネル数 x 盤面サイズ x 盤面サイズ] のテンソルとなる
    """
    channels = []

    for i in range(history_length):
        if i < len(history):
            board = history[-(i+1)]  # 最新の盤面から遡って取得
        else:
            board = np.zeros((board_size, board_size), dtype=np.float32)  # 履歴が足りない場合は空盤面
        # 黒石と白石をそれぞれ別チャネルに変換（1: 白か黒かのマスク）
        channels.append((board == 1).astype(np.float32))
        channels.append((board == 2).astype(np.float32))

    # 現在のプレイヤー情報（黒:1なら全マス1、白:2なら全マス0）
    current_plane = np.ones((board_size, board_size), dtype=np.float32) if current_player == 1 else np.zeros((board_size, board_size), dtype=np.float32)
    channels.append(current_plane)

    # 全チャネルをスタックして1つの配列にまとめる
    return np.stack(channels, axis=0)

def apply_dihedral_transform(input_array, transform_id):
    """
    盤面（または複数チャネルの配列）に対して、dihedral group（8通りの回転・反転）変換を適用する。
    transform_idが0〜3の場合は、回転のみ（90度単位）、
    4〜7の場合は左右反転後に回転
    """
    if transform_id < 4:
        return np.rot90(input_array, k=transform_id, axes=(1,2))
    else:
        flipped = np.flip(input_array, axis=2)  # 横反転
        return np.rot90(flipped, k=transform_id-4, axes=(1,2))

def transform_policy(target_policy, transform_id, board_size):
    """
    ターゲットポリシー（確率分布またはone-hot表現）に対して、dihedral変換を適用する関数。
    ・着手がパス（盤面の外部インデックス）なら変更しない。
    ・それ以外の場合、着手位置に対応するインデックスを回転・反転により変換する。
    """
    idx = np.argmax(target_policy)
    if idx == board_size * board_size:
        return target_policy  # パスの場合はそのまま
    row = idx // board_size
    col = idx % board_size

    # 単一の石が置かれた盤面を作成して変換
    board = np.zeros((board_size, board_size), dtype=np.float32)
    board[row, col] = 1.0
    transformed_board = apply_dihedral_transform(board[np.newaxis, ...], transform_id)[0]
    new_idx = np.argmax(transformed_board)
    new_policy = np.zeros_like(target_policy)
    new_policy[new_idx] = 1.0

    return new_policy

# ==============================
# 盤面クラス（囲碁盤の状態管理）
# ==============================
class Board:
    def __init__(self, size):
        self.size = size
        # 盤面状態をゼロ初期化（0: 空、1: 黒、2: 白）
        self.board = np.zeros((size, size), dtype=np.int8)

    def neighbors(self, row, col):
        """
        指定位置の上下左右の隣接マスを列挙する
        """
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            r, c = row+dr, col+dc
            if 0 <= r < self.size and 0 <= c < self.size:
                yield (r, c)

    def get_group(self, row, col):
        """
        与えられた位置から、同色の連（グループ）とそのグループが持つ呼吸点（隣接する空点）を取得する。
        深さ優先探索で実装。
        """
        color = self.board[row, col]
        group = []
        liberties = set()
        stack = [(row, col)]
        visited = set()
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            group.append((r, c))
            for nr, nc in self.neighbors(r, c):
                if self.board[nr, nc] == 0:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return group, liberties

    def play(self, move, color):
        """
        指定した場所に石を置き、相手石の捕獲や自分の石の自殺手の処理を行う。
        ・置く場所にすでに石があればエラーを投げる
        ・着手後、隣接している相手連の呼吸点を確認し、呼吸点がなければその連を盤面から除去
        ・その後、自分の連に呼吸点がなけければ自殺手として自身の連を除去
        """
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError("Illegal move: position already occupied")

        # color 'b'なら1, 'w'なら2として内部的に管理
        stone = 1 if color=='b' else 2
        self.board[row, col] = stone

        # 相手の石の色を決定
        opponent = 2 if stone==1 else 1

        # 隣接マスについて相手連をチェックし、呼吸点がなければ除去
        for nr, nc in self.neighbors(row, col):
            if self.board[nr, nc] == opponent:
                group, liberties = self.get_group(nr, nc)
                if len(liberties) == 0:
                    for r, c in group:
                        self.board[r, c] = 0

        # 自身が置いた石についても呼吸点がなければ取り除く（自殺手の検出）
        group, liberties = self.get_group(row, col)
        if len(liberties) == 0:
            for r, c in group:
                self.board[r, c] = 0

# ==============================
# Datasetクラス
# ==============================
class AlphaZeroSGFDatasetPreloaded(Dataset):
    """
    PyTorch用のDatasetクラス。前処理済みのSGFサンプルをメモリ上にロードしたもの。
    各サンプルは、盤面データ（flattenしたテンソル）、ターゲットポリシー、ターゲット値、マージンを含む。
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, pol, val, mar = self.samples[idx]
        # 盤面データをテンソル化し、ネットワーク入力の形状に変換
        board_tensor = torch.tensor(inp, dtype=torch.float32).view(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        target_policy_tensor = torch.tensor(pol, dtype=torch.float32)
        target_value_tensor = torch.tensor(val, dtype=torch.float32)
        target_margin_tensor = torch.tensor(mar, dtype=torch.float32)
        return board_tensor, target_policy_tensor, target_value_tensor, target_margin_tensor

# ==============================
# SGFからサンプル生成関数
# ==============================
def process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all):
    """
    1つのSGF文字列から、複数の学習サンプルを生成する関数。
    ・SGFファイルをパースして、各ノードごとに盤面履歴や着手情報を取り出す。
    ・データ拡張（dihedral変換）を適用できる（augment_allフラグ）
    ・最終的なサンプルは、flattenした盤面、one-hotなターゲットポリシー、ターゲット値、マージンの組み合わせ
    """
    samples = []
    try:
        sgf_data = parse_sgf(sgf_src)
    except Exception as e:
        sgf_logger.error(f"Error processing SGF text: {e}")
        return samples
    root = sgf_data["root"]

    try:
        # 盤面サイズはrootプロパティ「SZ」から取得。取得失敗時は引数のboard_sizeを使用
        sz = int(root.get(b'SZ')[0].decode('utf-8'))
    except Exception:
        sz = board_size

    # 勝敗情報：「RE」プロパティを取得。なければ"不明"
    result_prop = root.get(b'RE') if b'RE' in root else None
    result_str = result_prop[0].decode('utf-8') if result_prop and len(result_prop)>0 else "不明"

    # 勝敗の値として、黒勝ちなら1.0、白勝ちなら-1.0、引き分け等は0.0に設定
    target_value = 1.0 if result_str.startswith("B+") else -1.0 if result_str.startswith("W+") else 0.0
    try:
        # margin（点差等）の取得。失敗したら0.0
        target_margin = float(result_str[2:]) if result_str[2:] else 0.0
    except Exception:
        target_margin = 0.0

    board_obj = Board(sz)

    # 初期盤面（空盤）を履歴に追加
    history_boards = [board_obj.board.copy().astype(np.float32)]
    current_player = 1  # 初手は黒（1）から開始

    # SGFデータ内の各ノードを処理
    for node in sgf_data["nodes"]:

        # 現在のプレイヤーに応じて、着手プロパティ（B: 黒, W: 白）を決定
        move_prop = b'B' if current_player==1 else b'W'
        move_vals = node.get(move_prop)

        # 現在の盤面履歴からネットワークの入力テンソルを構築
        input_tensor = build_input_from_history(history_boards, current_player, sz, history_length)
        if move_vals is None or len(move_vals)==0 or move_vals[0]==b"":
            # 着手がない場合はパス（最後のインデックスを1に）
            target_policy = np.zeros(sz*sz+1, dtype=np.float32)
            target_policy[sz*sz] = 1.0
        else:
            try:
                # 着手情報のデコード。2文字（a～s等と仮定）で行インデックス、列インデックスを計算
                move = move_vals[0]
                col = ord(move.decode('utf-8')[0])-ord('a')
                row = ord(move.decode('utf-8')[1])-ord('a')
                target_policy = np.zeros(sz*sz+1, dtype=np.float32)
                target_policy[row*sz+col] = 1.0  # 対応するマスを1に
            except Exception as e:
                sgf_logger.warning(f"Error parsing move in SGF text: {e}")
                target_policy = np.zeros(sz*sz+1, dtype=np.float32)
                target_policy[sz*sz] = 1.0

        # データ拡張：augment_allがTrueなら8通り、Falseならランダム1通り
        transforms = range(8) if augment_all else [np.random.randint(0,8)]
        for t in transforms:
            inp = apply_dihedral_transform(input_tensor, t)
            pol = transform_policy(target_policy, t, sz)
            samples.append((
                inp.flatten(),  # flattenして1次元配列に変換
                pol,
                np.array([target_value], dtype=np.float32),
                np.array([target_margin], dtype=np.float32)
            ))

        # 着手が有る場合、盤面状態を更新し、履歴に追加
        if move_vals is not None and len(move_vals)>0 and move_vals[0]!=b"":
            try:
                move = move_vals[0]
                col = ord(move.decode('utf-8')[0])-ord('a')
                row = ord(move.decode('utf-8')[1])-ord('a')
                board_obj.play((row, col), 'b' if current_player==1 else 'w')
                history_boards.append(board_obj.board.copy().astype(np.float32))
            except Exception as e:
                sgf_logger.warning(f"Error updating board from SGF text: {e}")

        # プレイヤー交代
        current_player = 2 if current_player==1 else 1

    return samples

# ==============================
# 最良モデル保存用関数
# ==============================
def save_best_model(model, policy_accuracy, device, current_best_accuracy):
    """
    現在のPolicy Accuracyがこれまでの最高値を更新した場合、以下の処理を実施する：
      - 状態辞書形式でモデルを保存する
      - 推論専用モデル（TorchScript化）の保存
      - MODEL_OUTPUT_DIR内の精度が低いモデルファイルの削除
      - 新たな最高精度を返す
    """
    new_model_file = os.path.join(MODEL_OUTPUT_DIR, f"model_{policy_accuracy:.5f}.pt")
    # モデルの状態辞書を保存
    torch.save(model.state_dict(), new_model_file)
    train_logger.info(f"● New best model saved (state_dict): {new_model_file}")

    # 推論専用モデルのTorchScript化と保存
    save_inference_model(model, device, "inference2_model.pt")

    # モデル出力ディレクトリ内の他の低精度モデルファイルを削除
    for f in os.listdir(MODEL_OUTPUT_DIR):
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                acc = float(f[len("model_"):-len(".pt")])
                if acc < current_best_accuracy and os.path.join(MODEL_OUTPUT_DIR, f) != new_model_file:
                    os.remove(os.path.join(MODEL_OUTPUT_DIR, f))
                    train_logger.info(f"Deleted old model: {f}")
            except Exception:
                continue

    return max(policy_accuracy, current_best_accuracy)

# ==============================
# データセットの保存／読み込み関数
# ==============================
def save_dataset(samples, output_file):
    # pickle形式でサンプル群を保存
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)
    sgf_logger.info(f"Saved dataset to {output_file}")

def load_dataset(output_file):
    # pickle形式からサンプル群をロード
    with open(output_file, "rb") as f:
        samples = pickle.load(f)
    sgf_logger.info(f"Loaded dataset from {output_file}")
    return samples

# ==============================
# 推論専用モデル（TorchScript）保存関数
# ==============================
def save_inference_model(model, device, model_name):
    """
    モデルをCPUに移動して、torch.jit.traceを用いてTorchScript形式の推論専用モデルを生成・保存する。
    ・トレース用のダミー入力を用いて変換し、
    ・保存後、モデルを元のデバイスに戻す。
    """
    model_cpu = model.cpu()  # CPUへ移動
    dummy_input = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=torch.device("cpu"))
    traced_module = torch.jit.trace(model_cpu, dummy_input)
    inference_model_file = os.path.join(MODEL_OUTPUT_DIR, model_name)
    torch.jit.save(traced_module, inference_model_file)
    train_logger.info(f"Inference model saved as {inference_model_file}")
    model.to(device)  # 元のデバイスに戻す

# ==============================
# Test用データセット生成（zip利用）
# ==============================
def prepare_test_dataset(sgf_dir, board_size, history_length, augment_all, output_file):
    """
    テスト用のデータセットを生成する関数。
    ・既にpickleファイルが存在する場合はそれをロード。
    ・無ければ、SGFファイルからzipアーカイブを作成し、そこからサンプルを生成。
    ・生成したサンプルはpickle形式で保存する。
    """
    if os.path.exists(output_file):
        sgf_logger.info(f"Test dataset pickle {output_file} already exists. Loading it directly...")
        return load_dataset(output_file)

    if not os.path.exists(TEST_SGFS_ZIP):
        sgf_logger.info(f"Creating zip archive {TEST_SGFS_ZIP} from SGF files in {sgf_dir} ...")
        sgf_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir)
                     if f.endswith('.sgf') and "analyzed" not in f.lower()]
        with zipfile.ZipFile(TEST_SGFS_ZIP, 'w') as zf:
            for filepath in sgf_files:
                zf.write(filepath, arcname=os.path.basename(filepath))
        sgf_logger.info(f"Zip archive created: {TEST_SGFS_ZIP}")
    else:
        sgf_logger.info(f"Zip archive {TEST_SGFS_ZIP} already exists. Loading from it...")

    all_samples = []

    with zipfile.ZipFile(TEST_SGFS_ZIP, 'r') as zf:
        sgf_names = [name for name in zf.namelist() if name.endswith('.sgf') and "analyzed" not in name.lower()]
        sgf_names.sort()
        sgf_logger.info(f"TEST: Total SGF files in zip to process: {len(sgf_names)}")
        for name in tqdm(sgf_names, desc="Processing TEST SGF files"):
            try:
                sgf_src = zf.read(name).decode('utf-8')
                file_samples = process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all=False)
                all_samples.extend(file_samples)
            except Exception as e:
                sgf_logger.error(f"Error processing {name} from zip: {e}")

    save_dataset(all_samples, output_file)
    sgf_logger.info(f"TEST: Saved test dataset (total samples: {len(all_samples)}) to {output_file}")

    return all_samples

# ==============================
# グローバル変数：未処理のSGFファイルリスト
# ==============================
remaining_sgf_files = []

def prepare_train_dataset_cycle(sgf_dir, board_size, history_length, augment_all, max_files):
    """
    指定フォルダ内のSGFファイルから、1サイクル分の学習サンプルを生成する関数。
    ・全SGFファイルをランダム順に並び替え、max_files件分だけ処理する。
    ・ファイルごとにSGFテキストを読み込み、サンプルを生成する。
    """
    global remaining_sgf_files
    if not remaining_sgf_files:
        all_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir)
                     if f.endswith('.sgf') and "analyzed" not in f.lower()]
        random.shuffle(all_files)
        remaining_sgf_files = all_files
        sgf_logger.info("Regenerated the random order of all SGF files.")

    if len(remaining_sgf_files) < max_files:
        selected_files = remaining_sgf_files
        remaining_sgf_files = []  # 全部使い切る
        sgf_logger.info(f"Remaining SGF files less than max_files ({max_files}). Processing {len(selected_files)} files.")
    else:
        selected_files = remaining_sgf_files[:max_files]
        remaining_sgf_files = remaining_sgf_files[max_files:]
        sgf_logger.info(f"Selected {len(selected_files)} SGF files.")

    all_samples = []

    for sgf_file in selected_files:
        try:
            with open(sgf_file, "r", encoding="utf-8") as f:
                sgf_src = f.read()
            file_samples = process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all)
            all_samples.extend(file_samples)
        except Exception as e:
            sgf_logger.error(f"Error processing file {sgf_file}: {e}")

    random.shuffle(all_samples)
    sgf_logger.info(f"Training dataset cycle created. Total samples: {len(all_samples)}")

    return all_samples

def load_training_dataset(sgf_dir, board_size, history_length, augment_all, max_files):
    """
    トレーニング用のデータセットを一度だけ生成し、AlphaZeroSGFDatasetPreloadedのインスタンスとして返す。
    """
    samples = prepare_train_dataset_cycle(sgf_dir, board_size, history_length, augment_all, max_files)
    dataset = AlphaZeroSGFDatasetPreloaded(samples)

    return dataset

def validate_model(model, test_loader, device):
    """
    テスト用データセットを用いてモデルのpolicy accuracyを計算する関数。
    各バッチごとに、モデルの予測とターゲットのインデックスを比較し、正解数をカウントする。
    """
    model.eval()
    total_correct = 0
    total_samples_count = 0

    with torch.no_grad():
        for boards, target_policies, _, _ in tqdm(test_loader, desc="Validation", bar_format=bar_fmt):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            pred_policy, _ = model(boards)
            # 各サンプルについて、予測とターゲットの最大値のインデックスを比較
            correct = (pred_policy.argmax(dim=1) == target_policies.argmax(dim=1)).sum().item()
            total_correct += correct
            total_samples_count += boards.size(0)

    policy_accuracy = total_correct / total_samples_count
    train_logger.info(f"===== Validation Policy Accuracy ==== 【{policy_accuracy:.5f}】")

    return policy_accuracy

# ==============================
# 訓練ループ用関数（1エポック分）
# ==============================
def train_one_iteration(model, train_loader, optimizer, device):
    """
    1エポック分の訓練ループを実行する関数
    ・各バッチごとに損失の計算、逆伝播、パラメータ更新を行う
    ・policy loss, value loss, margin lossを各々計算して最終損失に加算する
    ・バッチごとの正解率も計算してログ出力する
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_margin_loss = 0.0
    num_batches = 0
    overall_correct = 0
    overall_samples = 0

    # 損失の重み（ハイパーパラメータ）
    value_loss_coefficient = 0.1
    margin_loss_coefficient = 0.0001

    print_interval = 100  # ログ出力するバッチ数の間隔
    accumulated_accuracy = 0.0
    group_batches = 0

    for boards, target_policies, target_values, target_margins in tqdm(train_loader, desc="Training", bar_format=bar_fmt):
        boards = boards.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        target_margins = target_margins.to(device)

        optimizer.zero_grad()  # 勾配の初期化
        pred_policy, (pred_value, pred_margin) = model(boards)

        # ポリシー損失は、ターゲットの対数確率との内積による負の和をバッチ数で割る
        policy_loss = -torch.sum(target_policies * pred_policy) / boards.size(0)

        # 値とマージンに対する平均二乗誤差
        value_loss = F.mse_loss(pred_value.view(-1), target_values.view(-1))
        margin_loss = F.mse_loss(pred_margin.view(-1), target_margins.view(-1))
        loss = policy_loss + value_loss_coefficient * value_loss + margin_loss_coefficient * margin_loss
        loss.backward()  # 逆伝播
        optimizer.step()  # パラメータ更新

        if USE_TPU:
            xm.mark_step()  # TPUの場合、明示的にステップをマークする必要がある

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_margin_loss += margin_loss.item()
        num_batches += 1

        # バッチごとに正解率（予測クラスとターゲットクラスの一致）を計算
        batch_pred = pred_policy.argmax(dim=1)
        batch_target = target_policies.argmax(dim=1)
        batch_accuracy = (batch_pred == batch_target).float().mean().item()
        overall_correct += (batch_pred == batch_target).sum().item()
        overall_samples += boards.size(0)
        accumulated_accuracy += batch_accuracy

        group_batches += 1

        if num_batches % print_interval == 0:
            avg_accuracy = accumulated_accuracy / group_batches
            start_batch = num_batches - group_batches + 1
            end_batch = num_batches
            print(f"Batch {start_batch:4d}～{end_batch:4d} policy accuracy average: {avg_accuracy:6.4f}")
            accumulated_accuracy = 0.0
            group_batches = 0

        del boards, target_policies, target_values, target_margins

    if group_batches > 0:
        avg_accuracy = accumulated_accuracy / group_batches
        print(f"Other ({group_batches} batch) policy accuracy average: {avg_accuracy:6.4f}")

    if overall_samples > 0:
        overall_accuracy = overall_correct / overall_samples
        print(f"Overall policy accuracy of the latest model state in this training loop: {overall_accuracy:6.4f}")
    else:
        overall_accuracy = 0.0

    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = value_loss_coefficient * total_value_loss / num_batches
    avg_margin_loss = margin_loss_coefficient * total_margin_loss / num_batches

    train_logger.info(f"Training iteration  total average loss: {avg_loss:.5f}")
    train_logger.info(f"Training iteration average policy loss: {avg_policy_loss:.5f}")
    train_logger.info(f"Training iteration  average value loss: {avg_value_loss:.5f}")
    train_logger.info(f"Training iteration average margin loss: {avg_margin_loss:.5f}")
    train_logger.info(f"Training iteration  overall p accuracy: {overall_accuracy:.5f}")

    return avg_loss

# ==============================
# チェックポイント保存＆復元
# ==============================
def save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_no_improve, best_policy_accuracy, checkpoint_file, device):
    """
    モデルの状態（パラメータやオプティマイザ情報）をチェックポイントとして保存する。
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'best_policy_accuracy': best_policy_accuracy
    }

    torch.save(checkpoint, checkpoint_file)

    train_logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_file}")

def recursive_to(data, device):
    """
    ネストされたデータ構造内の全てのtorch.Tensorを指定デバイスに移動するヘルパー関数
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: recursive_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to(item, device) for item in data]
    else:
        return data

def load_checkpoint(model, optimizer, checkpoint_file, device):
    """
    チェックポイントファイルからモデルとオプティマイザの状態を復元する。
    """
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

        # チェックポイント内の各パラメータを指定デバイスに移動
        new_state_dict = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_state_dict)

        optimizer_state = recursive_to(checkpoint['optimizer_state_dict'], device)
        optimizer.load_state_dict(optimizer_state)

        epoch = checkpoint['epoch']
        best_policy_accuracy = checkpoint.get('best_policy_accuracy', 0.0)

        train_logger.info(f"Checkpoint loaded from {checkpoint_file} at epoch {epoch}")

        return epoch, best_policy_accuracy
    else:
        train_logger.info("No checkpoint found. Starting from scratch.")
        return 0, 0.0

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
def _mp_fn(rank):
    """
    TPU分散環境（もしくはシングルプロセス）で実行されるメインの関数
    ・デバイスの設定、データセットの生成、モデルの構築、チェックポイントの復元、
    ・学習ループおよび評価処理を行う。
    """
    if USE_TPU:
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method='xla://')
        device = xm.xla_device()
        train_logger.info("Running on TPU device: {}".format(device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_logger.info("Running on device: {}".format(device))

    # テスト用データセットのpickleファイルのパス設定
    test_dataset_pickle = os.path.join(VAL_SGF_DIR, "test_dataset.pkl")

    # テスト用サンプルを生成またはロード
    test_samples = prepare_test_dataset(VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, True, test_dataset_pickle)
    test_dataset = AlphaZeroSGFDatasetPreloaded(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ネットワークのインスタンス生成し、指定デバイスへ移動
    model = EnhancedResNetPolicyValueNetwork(
        board_size=BOARD_SIZE,
        num_channels=model_channels,
        num_residual_blocks=num_residual_blocks,
        in_channels=NUM_CHANNELS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習率スケジューラの設定（評価指標が停滞した場合に減衰）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)

    # チェックポイントから復元
    start_epoch, best_policy_accuracy = load_checkpoint(model, optimizer, CHECKPOINT_FILE, device)

    train_logger.info("Initial best_policy_accuracy: {:.5f}".format(best_policy_accuracy))
    current_lr = optimizer.param_groups[0]['lr']
    train_logger.info("Current learning rate : {:.8f}".format(current_lr))

    # トレーニング用データセットの生成
    training_dataset = load_training_dataset(TRAIN_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, augment_all=True, max_files=number_max_files)

    epoch = start_epoch

    # 無限ループで学習・評価・チェックポイント保存を繰り返す
    while True:
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        train_one_iteration(model, train_loader, optimizer, device)
        epoch += 1
        policy_accuracy = validate_model(model, test_loader, device)
        if policy_accuracy > best_policy_accuracy:
            best_policy_accuracy = save_best_model(model, policy_accuracy, device, best_policy_accuracy)
        else:
            # 改良がなかった場合でも一時的な推論用モデルを保存
            save_inference_model(model, device, "inference2_model_tmp.pt")

        lr_before = optimizer.param_groups[0]['lr']
        train_logger.info("Epoch {} - Before scheduler.step(): lr = {:.8f}".format(epoch, lr_before))
        scheduler.step(policy_accuracy)
        lr_after = optimizer.param_groups[0]['lr']
        train_logger.info("Epoch {} - After scheduler.step(): lr = {:.8f}".format(epoch, lr_after))

        # ダミーの評価損失、エポック不改善回数（本実装では利用していない）を用いてチェックポイント保存
        dummy_best_val_loss = 0.0
        dummy_epochs_no_improve = 0
        save_checkpoint(model, optimizer, epoch, dummy_best_val_loss, dummy_epochs_no_improve, best_policy_accuracy, CHECKPOINT_FILE, device)
        train_logger.info("Iteration completed. Restarting next iteration...\n")

if __name__ == "__main__":
    # コマンドライン引数の設定（設定ファイルとチェックポイントファイルのパスを指定可能）
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config_py.ini"),
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_FILE,
                        help="Path to checkpoint file")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        sgf_logger.warning("Config file not found. Using default hyperparameters.")
    train_logger.info("=== Starting Training and Validation Loop ===")

    if USE_TPU:
        # TPU利用時は複数プロセス起動のため、xmp.spawnを利用（ここではnprocs=1）
        import torch_xla.distributed.xla_multiprocessing as xmp
        nprocs = 1
        xmp.spawn(_mp_fn, args=(), nprocs=nprocs)
    else:
        _mp_fn(0)

     