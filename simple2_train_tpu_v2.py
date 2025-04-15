#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
(詳細な説明は省略されていますが、全体としては囲碁棋譜（SGF）のパース、前処理、
ディープラーニングモデルの定義、学習ループ、チェックポイント保存・復元、及び
TPUやGoogle Colabを利用する際の環境設定などを行うコード)
"""

# ===== 固定定義：環境切り替え用フラグ =====
# TPUやGoogle Colab環境で動作させるかどうかのフラグ。環境に応じた挙動を切り替える
USE_TPU = True      # TPU を利用する場合は True にする
USE_COLAB = True    # Google Colab 環境で実行する場合は True にする

# ------------------------------
# 必要なライブラリのインポート
# ------------------------------
import os, re, pickle, zipfile, random, numpy as np, configparser, argparse
from tqdm import tqdm             # 進捗表示用ライブラリ（ループの状況を視覚的に表示）
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # データセットとデータローダのクラス
import torch.distributed as dist    # 分散学習に必要なモジュール
import time                         # 時間計測・待機処理に利用

# TPU利用時に必要なtorch_xla関連モジュールをインポート
if USE_TPU:
    import torch_xla
    import torch_xla.core.xla_model as xm         # TPUデバイスの取得に使用
    import torch_xla.distributed.xla_backend       # TPU向けの分散学習の設定

# Google Colab利用時にGoogle Driveをマウントする
if USE_COLAB:
    try:
        # すでにGoogle Driveがマウントされている場合、まずはアンマウントする試み（エラーは無視）
        os.system("fusermount -u /content/drive")
    except Exception as e:
        print("Google Drive unmount failed:", e)
    try:
        from google.colab import drive
        # Google Drive を強制的にリマウントする
        drive.mount('/content/drive', force_remount=True)
    except ImportError:
        print("Google Colab module not found.")

# ==============================
# デバイス設定
# ==============================
if USE_TPU:
    # TPUを利用する場合、torch_xlaを用いてTPUデバイスを取得
    device = xm.xla_device()
    # 分散処理のプロセスグループが未初期化なら、xlaプロトコルを用いて初期化
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method='xla://')
else:
    # TPUを使わない場合は、GPU（CUDA）が利用可能ならGPU、なければCPUを利用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# ディレクトリ設定
# ==============================
if USE_COLAB:
    # Google Colab環境向けのディレクトリ設定。Google Drive内の特定のフォルダを利用する
    BASE_DIR = "/content/drive/My Drive/sgf"              # ベースディレクトリ
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf_KK")  # 学習用SGFファイルのディレクトリ
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")            # 評価用SGFファイルのディレクトリ
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")# テスト用SGFファイル群をまとめたZIPファイル
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")      # モデル出力用ディレクトリ
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint2.pt")  # チェックポイントファイルのパス
else:
    # ローカル環境（例: Windows環境）用のディレクトリ設定
    BASE_DIR = r"D:\igo\simple2"
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf")
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint2.pt")

# モデル出力用のディレクトリが存在しない場合は自動で作成
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# ==============================
# tqdm の表示
# ==============================
# カスタムフォーマッタを定義（常に hh:mm:ss 形式で出力）
def fixed_format_interval(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

# tqdm の format_interval を上書き
tqdm.format_interval = fixed_format_interval

# プログレスバーのフォーマットを指定（tqdmで利用）
bar_fmt = "{l_bar}{bar}| {n:>6d}/{total:>6d} [{elapsed}<{remaining}, {rate_fmt}]"

# ==============================
# DummyLogger クラス（ログ出力用）
# ==============================
from datetime import datetime, timedelta, timezone
# JST（日本標準時）のタイムゾーン設定（UTC+9）
JST = timezone(timedelta(hours=9), 'JST')

class DummyLogger:
    # infoレベルのログ出力関数。時刻付きでメッセージを標準出力する
    def info(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} INFO: {message}", *args, **kwargs)

    # warningレベルのログ出力関数
    def warning(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} WARNING: {message}", *args, **kwargs)

    # errorレベルのログ出力関数
    def error(self, message, *args, **kwargs):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} ERROR: {message}", *args, **kwargs)

# ログ出力用のインスタンスを生成（SGF処理用と学習用でそれぞれ利用）
sgf_logger = DummyLogger()
train_logger = DummyLogger()

# ==============================
# 設定ファイル読み込み関数
# ==============================
def load_config(config_path):
    """
    指定した設定ファイル（INI形式）から各パラメータを読み込む関数

    Parameters:
        config_path (str): 設定ファイルのパス

    Returns:
        dict: 取得した各パラメータをキーに持つ辞書
    """
    config = configparser.ConfigParser()  # ConfigParserオブジェクトの生成
    config.read(config_path)             # ファイルの読み込み

    try:
        # BOARDセクション：盤のサイズ（例：19x19）を取得。指定がなければ"19"を使用
        BOARD_SIZE = int(config.get("BOARD", "board_size", fallback="19"))
        # DATAセクション：履歴の長さを取得。なければ"8"を使用
        HISTORY_LENGTH = int(config.get("DATA", "history_length", fallback="8"))
        # 入力チャネル数は、各履歴で2チャネル（黒、白）＋1チャネル（現在のプレイヤー情報）
        NUM_CHANNELS = 2 * HISTORY_LENGTH + 1
        # 行動の総数は、盤面のマス数＋パス用1手
        NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1

        # MODELセクション：残差ブロックの数とチャネル数（中間層のフィルタ数）を取得
        num_residual_blocks = int(config.get("MODEL", "num_residual_blocks", fallback="20"))
        model_channels = int(config.get("MODEL", "model_channels", fallback="256"))

        # TRAINセクション：エポック数、バッチサイズ、学習率、patience、及び学習率減衰率factorを取得
        num_epochs = int(config.get("TRAIN", "num_epochs", fallback="100"))
        batch_size = int(config.get("TRAIN", "batch_size", fallback="256"))
        learning_rate = float(config.get("TRAIN", "learning_rate", fallback="0.001"))
        patience = int(config.get("TRAIN", "patience", fallback="10"))
        factor = float(config.get("TRAIN", "factor", fallback="0.8"))
        # 一回のデータセット生成時に処理するファイル数
        number_max_files = int(config.get("TRAIN", "number_max_files", fallback="256"))
    except Exception as e:
        # 読み込みエラーが発生した場合はエラーログを出力し、プログラムを終了する
        train_logger.error(f"Error reading configuration: {e}")
        exit(1)

    # 各パラメータを辞書形式で返す
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

# 設定ファイルのパス（BASE_DIR配下にあると仮定）
CONFIG_PATH = os.path.join(BASE_DIR, "config_py.ini")
# 設定ファイルを読み込み、各パラメータを辞書に格納
config_params = load_config(CONFIG_PATH)

# ロードしたパラメータをグローバル変数に展開する
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

# ログ出力で設定内容を確認
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
# ResidualBlock : 標準的な残差ブロック（スキップ接続を含む）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        パラメータ:
          channels (int): 入力および出力のチャネル数
        """
        super(ResidualBlock, self).__init__()
        # カーネルサイズ3、パディング1の畳み込み層を2層定義
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # 畳み込み後のバッチ正規化層
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        # 入力を残差として保持（スキップ接続）
        residual = x
        # 1回目の畳み込み→正規化→ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 2回目の畳み込み→正規化
        out = self.bn2(self.conv2(out))
        # スキップ接続により、入力と出力を加算
        out += residual
        # 最終的にReLUを適用して出力
        return F.relu(out)

# DilatedResidualBlock : 拡張（dilated）畳み込みを用いた残差ブロック
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        """
        パラメータ:
          channels (int): 入力および出力チャネル数
          dilation (int): 拡張率（dilation rate）
        """
        super(DilatedResidualBlock, self).__init__()
        # 拡張畳み込み（dilation指定、パディングにdilationの値を利用）
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        # 通常の畳み込み層
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x  # 入力をスキップ接続用に保持
        out = F.relu(self.bn1(self.conv1(x)))  # 拡張畳み込み＋正規化＋ReLU
        out = self.bn2(self.conv2(out))         # 通常の畳み込み＋正規化
        out += residual                         # スキップ接続：入力を加算
        return F.relu(out)                      # 最後にReLUを適用して出力

# SelfAttention : セルフアテンション機構の層。空間的な自己相関を計算する
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        """
        パラメータ:
          in_channels (int): 入力チャネル数
        """
        super(SelfAttention, self).__init__()
        # クエリを得るための1x1畳み込み。チャネル数を1/8に縮小
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # キーを得るための1x1畳み込み（同じくチャネルを1/8に）
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # バリューを得るための1x1畳み込み（チャネル数は変えない）
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 出力へのスケールパラメータ（学習可能なパラメータとして0で初期化）
        self.gamma = nn.Parameter(torch.zeros(1))
        # ソフトマックス関数でアテンション係数を正規化（最後の次元で適用）
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        batch, C, H, W = x.size()  # 入力の形状を取得
        # クエリを計算し、形状を (batch, H*W, C//8) に変形
        proj_query = self.query_conv(x).view(batch, -1, H * W).permute(0, 2, 1)
        # キーを計算し、形状を (batch, C//8, H*W) に変形
        proj_key = self.key_conv(x).view(batch, -1, H * W)
        # クエリとキーのバッチ積により相関（エネルギー）を計算
        energy = torch.bmm(proj_query, proj_key)
        # ソフトマックスにより、各位置での注意重みを計算
        attention = self.softmax(energy)
        # バリューを計算し、形状を (batch, C, H*W) に変形
        proj_value = self.value_conv(x).view(batch, -1, H * W)
        # バッチ積により、注目情報を反映した出力を計算（注意の転置を使用）
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)  # 元の空間サイズに戻す
        # スケールパラメータを掛けた出力を元入力に足し込み、最終出力とする
        return self.gamma * out + x

# EnhancedResNetPolicyValueNetwork : 改良版残差ネットワークに基づく、ポリシーとバリューを同時に出力するネットワーク
class EnhancedResNetPolicyValueNetwork(nn.Module):
    def __init__(self, board_size, num_channels, num_residual_blocks, in_channels):
        """
        パラメータ:
          board_size (int): 盤面のサイズ（例：19）
          num_channels (int): 入力特徴マップのチャネル数
          num_residual_blocks (int): 残差ブロックの数
          in_channels (int): ネットワークへの入力チャネル数
        """
        super(EnhancedResNetPolicyValueNetwork, self).__init__()
        self.board_size = board_size
        # 入力層：3x3畳み込みとバッチ正規化を適用して特徴抽出
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        # 残差ブロックをリストに格納。一定ごとに拡張畳み込みブロックを使用
        blocks = []
        for i in range(num_residual_blocks):
            if i % 4 == 0:
                blocks.append(DilatedResidualBlock(num_channels, dilation=2))
            else:
                blocks.append(ResidualBlock(num_channels))
        # リストのブロックを連結してシーケンシャルなモデルにする
        self.residual_blocks = nn.Sequential(*blocks)
        # セルフアテンション層を配置（特徴マップ内の依存関係を補正）
        self.attention = SelfAttention(num_channels)
        # ----- ポリシーヘッドの構築 -----
        # 1x1畳み込みとバッチ正規化でチャネル数を2に縮小
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.dropout_policy = nn.Dropout(p=0.5)  # 過学習防止のためDropoutを適用
        # 全結合層：2チャネルの各位置をフラット化し、全ての行動（盤面のマス＋パス）を出力
        self.fc_policy = nn.Linear(2 * board_size * board_size, NUM_ACTIONS)
        # ----- バリューヘッドの構築 -----
        # 1x1畳み込みとバッチ正規化で1チャネルに縮小し、盤面全体の特徴を抽出
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        # 隠れ層全結合：盤面情報を64次元へ変換
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.dropout_value = nn.Dropout(p=0.5)
        # 最終出力は2ユニット。1つはvalue（勝ち負けの評価）、もう1つはmargin（差分情報）として出力
        self.fc_value2 = nn.Linear(64, 2)
    def forward(self, x):
        # 入力層処理：畳み込み→正規化→ReLU
        x = F.relu(self.bn_input(self.conv_input(x)))
        # 複数の残差ブロック（および拡張畳み込みブロック）の適用
        x = self.residual_blocks(x)
        # セルフアテンション層を適用して特徴マップを補正
        x = self.attention(x)
        # ---- ポリシーヘッド処理 ----
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = self.dropout_policy(p)
        p = p.view(p.size(0), -1)  # フラット化して全結合層への入力形状に
        p = self.fc_policy(p)
        p = F.log_softmax(p, dim=1)  # 出力の対数確率に変換
        # ---- バリューヘッド処理 ----
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = self.dropout_value(v)
        out = self.fc_value2(v)
        # 1ユニット目を [tanh] で正規化し[-1,1]の範囲に収め、残りをmarginとしてそのまま出力
        value = torch.tanh(out[:, 0])
        margin = out[:, 1]
        return p, (value, margin)

# ==============================
# SGFパーサー＆前処理関数
# ==============================
def parse_sgf(sgf_text):
    """
    SGF形式の棋譜文字列をパースして、棋譜のノード情報を抽出する関数。
    ・余分な空白や括弧を除去し、セミコロン毎にノードに分割
    ・各ノード内でプロパティ（例: B[dd], SZ[19]）を正規表現で抽出する
    """
    sgf_text = sgf_text.strip()  # 余分な空白除去
    if sgf_text.startswith('(') and sgf_text.endswith(')'):
        sgf_text = sgf_text[1:-1]  # 最初と最後の括弧を除去

    # セミコロンで分割し、空文字列でない部分のみ抽出
    parts = [part for part in sgf_text.split(';') if part.strip()]
    nodes = []

    # 正規表現パターン：大文字アルファベット＋角括弧内の値（複数文字対応）
    prop_pattern = re.compile(r'([A-Z]+)\[([^\]]*)\]')
    for part in parts:
        props = {}
        # 各プロパティ（キーと値）をループで抽出し、キーはUTF-8エンコードして利用
        for m in prop_pattern.finditer(part):
            key = m.group(1).encode('utf-8')
            value = m.group(2)
            props[key] = [value.encode('utf-8')]
        nodes.append(props)
    if not nodes:
        # もし抽出できるノードが1つも無ければ例外を発生
        raise ValueError("No nodes found in SGF file")
    # 最初のノードをroot、それ以外をノードリストとして返す
    return {"root": nodes[0], "nodes": nodes[1:]}

def build_input_from_history(history, current_player, board_size, history_length):
    """
    棋譜の履歴情報から、ディープラーニングモデルへの入力テンソルを作成する。
    ・各履歴盤面について、黒石と白石のプレゼンテーションを別チャネルに展開
    ・履歴が足りない場合は空盤面（ゼロ行列）を用いる
    ・さらに、現在のプレイヤー情報を1チャネル追加する
    戻り値は [チャネル数 x 盤面サイズ x 盤面サイズ] の3次元配列
    """
    channels = []

    for i in range(history_length):
        if i < len(history):
            board = history[-(i+1)]  # 最新の盤面から順に取得
        else:
            board = np.zeros((board_size, board_size), dtype=np.float32)  # 履歴が足りなければ空盤面
        # 黒石がある位置かどうかを判定して float32 のマスクを作成（1か0）
        channels.append((board == 1).astype(np.float32))
        # 同様に白石についてもチャネルとして追加
        channels.append((board == 2).astype(np.float32))

    # 現在のプレイヤー情報を追加。現在のプレイヤーが黒なら全1のマトリクス、白なら全0
    current_plane = np.ones((board_size, board_size), dtype=np.float32) if current_player == 1 else np.zeros((board_size, board_size), dtype=np.float32)
    channels.append(current_plane)

    # 各チャネルをスタックして 3 次元のnumpy配列にする
    return np.stack(channels, axis=0)

def apply_dihedral_transform(input_array, transform_id):
    """
    盤面や多チャネル配列に対して、dihedral groupの8通りの幾何学的変換（回転・反転）を適用する。
    transform_id が 0〜3 なら回転のみ（90度単位）、4〜7 なら左右反転後に回転を適用する。
    """
    if transform_id < 4:
        # np.rot90で指定回数回転（axes引数で回転軸を指定）
        return np.rot90(input_array, k=transform_id, axes=(1,2))
    else:
        # 横方向に反転してから、回転を適用
        flipped = np.flip(input_array, axis=2)
        return np.rot90(flipped, k=transform_id-4, axes=(1,2))

def transform_policy(target_policy, transform_id, board_size):
    """
    ターゲットポリシー（one-hotまたは確率分布）の変換。
    ・パス（最後のインデックス）が選ばれている場合は変換せずそのまま返す。
    ・その他の場合、インデックスを盤面上の行・列に変換してから、dihedral変換を適用し、
      新しいインデックスのone-hotベクトルを生成する。
    """
    idx = np.argmax(target_policy)
    if idx == board_size * board_size:
        # パスの場合は変更せず返す
        return target_policy
    row = idx // board_size
    col = idx % board_size

    # 対象のマスに1が立つ盤面（one-hot）を作成し、dihedral変換を適用する
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
        """
        パラメータ:
          size (int): 盤面のサイズ（例: 19）
        """
        self.size = size
        # 盤面は NumPy の int8 型の2次元配列。0: 空、1: 黒、2: 白
        self.board = np.zeros((size, size), dtype=np.int8)

    def neighbors(self, row, col):
        """
        指定した位置 (row, col) の上下左右の隣接座標をジェネレータで返す
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            # 座標が盤面内に収まる場合のみ yield する
            if 0 <= r < self.size and 0 <= c < self.size:
                yield (r, c)

    def get_group(self, row, col):
        """
        指定位置の石に連なる同色のグループと、そのグループが持つ呼吸点（空点）の集合を返す。
        深さ優先探索によりグループを探索する。
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
            # 隣接するマスそれぞれについて
            for nr, nc in self.neighbors(r, c):
                if self.board[nr, nc] == 0:
                    liberties.add((nr, nc))  # 空点なら呼吸点として追加
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    # 同色で未探索の場合、探索対象に追加
                    stack.append((nr, nc))
        return group, liberties

    def play(self, move, color):
        """
        指定した座標に石を置く関数。着手の合法性、相手石の捕獲、自殺手の検出を行う。
        Parameters:
          move (tuple): (row, col) のタプル
          color (str): 'b' (黒) もしくは 'w' (白)
        """
        row, col = move
        if self.board[row, col] != 0:
            # すでに石がある場合はエラーを投げる
            raise ValueError("Illegal move: position already occupied")
        # 内部的には、黒は1、白は2とする
        stone = 1 if color == 'b' else 2
        self.board[row, col] = stone

        # 相手の石の色を決定
        opponent = 2 if stone == 1 else 1

        # 隣接マスにある相手の石について、そのグループの呼吸点をチェック
        for nr, nc in self.neighbors(row, col):
            if self.board[nr, nc] == opponent:
                group, liberties = self.get_group(nr, nc)
                if len(liberties) == 0:
                    # もし呼吸点がなければ、グループ全体を盤面から除去（捕獲）
                    for r, c in group:
                        self.board[r, c] = 0

        # 自分の石のグループについて呼吸点が無ければ（自殺手の場合）、グループを除去する
        group, liberties = self.get_group(row, col)
        if len(liberties) == 0:
            for r, c in group:
                self.board[r, c] = 0

# ==============================
# Datasetクラス（PyTorch用）
# ==============================
class AlphaZeroSGFDatasetPreloaded(Dataset):
    """
    前処理済みのSGFサンプルをメモリ上に保持するDatasetクラス。
    各サンプルは、flattenした盤面の配列、ターゲットポリシー、ターゲット値、ターゲットマージンからなる。
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # サンプルは (入力, ポリシー, 値, マージン) の順番となっている
        inp, pol, val, mar = self.samples[idx]
        # 入力盤面をテンソルに変換し、[チャネル数 x 盤面サイズ x 盤面サイズ]の形状にリシェイプ
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
    1つのSGF文字列から複数の学習サンプルを生成する関数。
    ・SGF文字列をパースし、棋譜の各ノードごとに盤面の履歴を取得
    ・各ノードに対して、入力テンソル、ターゲットポリシー、ターゲット値、マージンのサンプルを生成
    ・augment_all フラグがTrueなら8通り（dihedral変換）でデータ拡張を行う
    """
    samples = []
    try:
        sgf_data = parse_sgf(sgf_src)
    except Exception as e:
        sgf_logger.error(f"Error processing SGF text: {e}")
        return samples
    root = sgf_data["root"]

    try:
        # 盤面サイズは、rootプロパティ「SZ」から取得。取得に失敗すれば引数のboard_sizeを使用
        sz = int(root.get(b'SZ')[0].decode('utf-8'))
    except Exception:
        sz = board_size

    # 勝敗情報を示すプロパティ「RE」を取得。なければ"不明"とする
    result_prop = root.get(b'RE') if b'RE' in root else None
    result_str = result_prop[0].decode('utf-8') if result_prop and len(result_prop) > 0 else "不明"

    # 勝敗の値として、黒勝ちなら 1.0、白勝ちなら -1.0、引き分け等は 0.0 とする
    target_value = 1.0 if result_str.startswith("B+") else -1.0 if result_str.startswith("W+") else 0.0
    try:
        # マージン（差分情報）を取得。解析できなければ 0.0 を使用
        target_margin = float(result_str[2:]) if result_str[2:] else 0.0
    except Exception:
        target_margin = 0.0

    # Board クラスのインスタンスを作成（盤面の初期状態は空盤）
    board_obj = Board(sz)
    # 初期盤面（空盤）を履歴に追加
    history_boards = [board_obj.board.copy().astype(np.float32)]
    current_player = 1  # 初手は黒（1）が始める

    # SGF内の各ノードを順次処理する
    for node in sgf_data["nodes"]:
        # 現在のプレイヤーに応じて、着手プロパティを決定
        move_prop = b'B' if current_player == 1 else b'W'
        move_vals = node.get(move_prop)
        # 現在の盤面履歴からネットワーク入力となるテンソルを作成
        input_tensor = build_input_from_history(history_boards, current_player, sz, history_length)
        if move_vals is None or len(move_vals) == 0 or move_vals[0] == b"":
            # 着手情報が無い場合は、パスとして扱い、盤面外を示すインデックスを1にする
            target_policy = np.zeros(sz * sz + 1, dtype=np.float32)
            target_policy[sz * sz] = 1.0
        else:
            try:
                # 着手情報のデコード処理。1文字目が列、2文字目が行（'a'～など）で指定される
                move = move_vals[0]
                col = ord(move.decode('utf-8')[0]) - ord('a')
                row = ord(move.decode('utf-8')[1]) - ord('a')
                target_policy = np.zeros(sz * sz + 1, dtype=np.float32)
                target_policy[row * sz + col] = 1.0  # 該当マスを1に設定
            except Exception as e:
                sgf_logger.warning(f"Error parsing move in SGF text: {e}")
                target_policy = np.zeros(sz * sz + 1, dtype=np.float32)
                target_policy[sz * sz] = 1.0

        # データ拡張処理（dihedral変換）：augment_allがTrueなら8通り、Falseならランダムに1通りを適用
        transforms = range(8) if augment_all else [np.random.randint(0, 8)]
        for t in transforms:
            inp = apply_dihedral_transform(input_tensor, t)
            pol = transform_policy(target_policy, t, sz)
            samples.append((
                inp.flatten(),  # 盤面入力を1次元にフラット化
                pol,            # 変換後のターゲットポリシー
                np.array([target_value], dtype=np.float32),   # ターゲット値
                np.array([target_margin], dtype=np.float32)     # ターゲットマージン
            ))
        # 着手がある場合、盤面状態を更新して履歴に追加する
        if move_vals is not None and len(move_vals) > 0 and move_vals[0] != b"":
            try:
                move = move_vals[0]
                col = ord(move.decode('utf-8')[0]) - ord('a')
                row = ord(move.decode('utf-8')[1]) - ord('a')
                # Boardクラスのplayメソッドを用いて石を置く（黒の場合'b'、白の場合'w'）
                board_obj.play((row, col), 'b' if current_player == 1 else 'w')
                # 盤面状態をコピーして履歴リストに追加
                history_boards.append(board_obj.board.copy().astype(np.float32))
            except Exception as e:
                sgf_logger.warning(f"Error updating board from SGF text: {e}")
        # プレイヤーの交代：黒→白、白→黒
        current_player = 2 if current_player == 1 else 1

    return samples

# ==============================
# データセットの保存／読み込み関数
# ==============================
def save_dataset(samples, output_file):
    """
    サンプル群をpickle形式でファイルに保存する関数
    """
    with open(output_file, "wb") as f:
        pickle.dump(samples, f)
    sgf_logger.info(f"Saved dataset to {output_file}")

def load_dataset(output_file):
    """
    pickleファイルからサンプル群を読み込む関数
    """
    with open(output_file, "rb") as f:
        samples = pickle.load(f)
    sgf_logger.info(f"Loaded dataset from {output_file}")
    return samples

# ==============================
# 推論専用モデル（TorchScript）保存関数
# ==============================
def save_inference_model(model, device, model_name):
    """
    モデルをCPUに移動し、torch.jit.traceを用いてTorchScript形式の推論専用モデルを生成・保存する関数。
    保存後、元のデバイスにモデルを戻す。
    """
    model_cpu = model.cpu()   # モデルをCPUに移動
    # 推論のトレース用ダミー入力（形状は [1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE]）
    dummy_input = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=torch.device("cpu"))
    traced_module = torch.jit.trace(model_cpu, dummy_input)  # TorchScript形式に変換
    inference_model_file = os.path.join(MODEL_OUTPUT_DIR, model_name)
    torch.jit.save(traced_module, inference_model_file)  # ファイルに保存
    train_logger.info(f"Inference model saved as {inference_model_file}")
    model.to(device)  # 元のデバイスに戻す

# ==============================
# 最良モデル保存用関数
# ==============================
def save_best_model(model, policy_accuracy, device, current_best_accuracy):
    """
    現在のpolicy accuracyがこれまでの最高を更新した場合に以下を行う：
      - モデルの状態辞書をファイルに保存
      - 推論専用モデル（TorchScript）の保存
      - MODEL_OUTPUT_DIR内で低精度のモデルファイルの削除
      - 最新の最高値を返す
    """
    new_model_file = os.path.join(MODEL_OUTPUT_DIR, f"model_{policy_accuracy:.5f}.pt")

    # モデル状態を保存
    torch.save(model.state_dict(), new_model_file)
    train_logger.info(f"● New best model saved (state_dict): {new_model_file}")

    # 推論専用モデルの保存
    save_inference_model(model, device, "inference2_model.pt")

    # 既存の低精度モデルファイルを削除する
    for f in os.listdir(MODEL_OUTPUT_DIR):
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                acc = float(f[len("model_"):-len(".pt")])
                if acc < current_best_accuracy and os.path.join(MODEL_OUTPUT_DIR, f) != new_model_file:
                    os.remove(os.path.join(MODEL_OUTPUT_DIR, f))
                    train_logger.info(f"Deleted old model: {f}")
            except Exception:
                continue

    # 現在の最高値と比較して、大きい方を返す
    return max(policy_accuracy, current_best_accuracy)

# ==============================
# モデル検証関数
# ==============================
def validate_model(model, test_loader, device):
    """
    テスト用データセットを用いて、モデルのpolicy accuracyを計算する関数。
    各バッチごとに予測とターゲットの最大値インデックスの一致数をカウントし、全体の正解率を算出する。
    """
    model.eval()  # 評価モードに切り替え
    total_correct = 0
    total_samples_count = 0

    with torch.no_grad():  # 評価時は勾配計算を行わない
        for boards, target_policies, _, _ in tqdm(test_loader, desc="Validation", bar_format=bar_fmt):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            pred_policy, _ = model(boards)
            # 各サンプルで、最も高い確率のインデックスが一致するか判定
            correct = (pred_policy.argmax(dim=1) == target_policies.argmax(dim=1)).sum().item()
            total_correct += correct
            total_samples_count += boards.size(0)

    policy_accuracy = total_correct / total_samples_count
    train_logger.info(f"===== Validation Policy Accuracy ==== 【{policy_accuracy:.5f}】")
    return policy_accuracy

# ==============================
# チェックポイント保存＆復元関数
# ==============================
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve, best_policy_accuracy, checkpoint_file, device):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # スケジューラ状態の保存
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'best_policy_accuracy': best_policy_accuracy
    }
    torch.save(checkpoint, checkpoint_file)
    sgf_logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_file}")

# 訓練中のコンソール表示が崩れないように、ログ出力はしないバージョン
def save_checkpoint_nolog(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve, best_policy_accuracy, checkpoint_file, device):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # スケジューラ状態の保存
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'best_policy_accuracy': best_policy_accuracy
    }
    torch.save(checkpoint, checkpoint_file)

def recursive_to(data, device):
    """
    ネストされたデータ（辞書、リスト等）の中にあるtorch.Tensorをすべて指定デバイスに移動するヘルパー関数
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: recursive_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to(item, device) for item in data]
    else:
        return data

def load_checkpoint(model, optimizer, scheduler, checkpoint_file, device):
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        new_state_dict = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_state_dict)
        optimizer_state = recursive_to(checkpoint['optimizer_state_dict'], device)
        optimizer.load_state_dict(optimizer_state)

        # スケジューラ状態の復元を追加
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_policy_accuracy = checkpoint.get('best_policy_accuracy', 0.0)
        train_logger.info(f"Checkpoint loaded from {checkpoint_file} at epoch {epoch}")
        return epoch, best_policy_accuracy
    else:
        train_logger.info("No checkpoint found. Starting from scratch.")
        return 0, 0.0

# ==============================
# sgfファイル用進捗チェックポイント保存＆復元
# ==============================
# チェックポイントファイルの保存先を指定
PROGRESS_CHECKPOINT_FILE = os.path.join(BASE_DIR, "progress_checkpoint.pkl")

def save_progress_checkpoint(remaining_files):
    """
    残りのSGFファイルリストをpickle形式で保存する関数
    """
    with open(PROGRESS_CHECKPOINT_FILE, "wb") as f:
        pickle.dump(remaining_files, f)
    sgf_logger.info(f"Progress checkpoint saved to {PROGRESS_CHECKPOINT_FILE}")

def load_progress_checkpoint():
    """
    進捗チェックポイントファイルから、残りのSGFファイルリストを読み込む関数
    """
    if os.path.exists(PROGRESS_CHECKPOINT_FILE):
        with open(PROGRESS_CHECKPOINT_FILE, "rb") as f:
            remaining_files = pickle.load(f)
        sgf_logger.info(f"Progress checkpoint loaded from {PROGRESS_CHECKPOINT_FILE}")
        return remaining_files
    else:
        sgf_logger.info("No progress checkpoint found.")
        return None

# ==============================
# Test用データセット生成（zip利用）
# ==============================
def prepare_test_dataset(sgf_dir, board_size, history_length, augment_all, output_file):
    """
    テスト用データセットを生成する関数
    ・既にpickleファイルが存在する場合はそれをロードし、無ければSGFファイルからzipを作成してサンプル生成
    ・生成したサンプルはpickle形式で保存する
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
        # SGFファイルの名前リストを取得
        sgf_names = [name for name in zf.namelist() if name.endswith('.sgf') and "analyzed" not in name.lower()]
        sgf_names.sort()  # 名前順にソート
        sgf_logger.info(f"TEST: Total SGF files in zip to process: {len(sgf_names)}")
        for name in tqdm(sgf_names, desc="Processing TEST SGF files"):
            try:
                sgf_src = zf.read(name).decode('utf-8')
                file_samples = process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all=False)
                all_samples.extend(file_samples)
            except Exception as e:
                sgf_logger.error(f"Error processing {name} from zip: {e}")

    # 生成されたサンプルをpickleファイルに保存
    save_dataset(all_samples, output_file)
    sgf_logger.info(f"TEST: Saved test dataset (total samples: {len(all_samples)}) to {output_file}")

    return all_samples

# ==============================
# 訓練データ生成関数
# ==============================
# グローバル変数（未処理のSGFファイルリスト）
remaining_sgf_files = []

def prepare_train_dataset_cycle(sgf_dir, board_size, history_length, resume_flag, augment_all, max_files):
    """
    指定フォルダ内のSGFファイルから、1サイクル分の学習サンプルを生成する関数。
    ・resume_flag が True の場合、進捗チェックポイントから残りファイルリストを読み込む。
    ・ただし、プログラム起動直後は FORCE_RELOAD が True なので、常に全件再読み込みし、
      その後 FORCE_RELOAD を False にすることで、以降は前回の進捗チェックポイントを利用する。
    ・ファイル全体をランダムな順序に並べ替え、max_files 件分だけ処理する。
    ・処理後、残りファイルリストの進捗を保存する。
    """
    global remaining_sgf_files, FORCE_RELOAD

    # resume_flag が True かつ FORCE_RELOAD が False の場合のみ、進捗チェックポイントからロード
    if resume_flag and not FORCE_RELOAD:
        remaining = load_progress_checkpoint()
        if remaining is not None:
            remaining_sgf_files = remaining

    # FORCE_RELOAD が True または remaining_sgf_files が空の場合、全ファイルを再読み込みする
    if FORCE_RELOAD or not remaining_sgf_files:
        all_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir)
                     if f.endswith('.sgf') and "analyzed" not in f.lower()]
        random.shuffle(all_files)
        remaining_sgf_files = all_files
        sgf_logger.info(f"Regenerated the random order of all SGF files : {len(all_files)} (FORCE_RELOAD was {FORCE_RELOAD})")
        # FORCE_RELOAD のフラグは一度強制再読み込みを行ったら False にする
        FORCE_RELOAD = False

    # remaining_sgf_files が max_files に満たない場合は全件、満たす場合は先頭 max_files 件を選択
    if len(remaining_sgf_files) < max_files:
        selected_files = remaining_sgf_files
        remaining_sgf_files = []  # リストを空にする
        sgf_logger.info(f"Remaining SGF files less than max_files ({max_files}). Processing {len(selected_files)} files.")
    else:
        selected_files = remaining_sgf_files[:max_files]
        remaining_sgf_files = remaining_sgf_files[max_files:]
        sgf_logger.info(f"Selected {len(selected_files)} SGF files. Remaining files : {len(remaining_sgf_files)}")

    all_samples = []
    # 選択した各SGFファイルについてサンプル生成を実施
    for sgf_file in selected_files:
        try:
            with open(sgf_file, "r", encoding="utf-8") as f:
                sgf_src = f.read()
            file_samples = process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all)
            all_samples.extend(file_samples)
        except Exception as e:
            sgf_logger.error(f"Error processing file {sgf_file}: {e}")

    random.shuffle(all_samples)  # 学習サンプルをシャッフル
    sgf_logger.info(f"Training dataset cycle created. Total samples: {len(all_samples)}")

    # 進捗チェックポイントとして、残りのSGFファイルリストを保存
    save_progress_checkpoint(remaining_sgf_files)

    return all_samples

def load_training_dataset(sgf_dir, board_size, history_length, resume_flag, augment_all, max_files):
    """
    トレーニング用のデータセットを一度だけ生成し、
    AlphaZeroSGFDatasetPreloaded のインスタンスとして返す関数。
    """
    samples = prepare_train_dataset_cycle(sgf_dir, board_size, history_length, resume_flag, augment_all, max_files)
    dataset = AlphaZeroSGFDatasetPreloaded(samples)

    return dataset

# ==============================
# 訓練ループ用関数（1エポック分）
# ==============================
def train_one_iteration(model, train_loader, optimizer, scheduler, device,  epoch, checkpoint_interval, best_policy_accuracy):
    """
    1エポック分の訓練ループを実行する関数
    ・各バッチごとに、入力に対する損失（policy loss, value loss, margin loss）を計算し、逆伝播を実行する。
    ・バッチごとの正解率も計算し、一定バッチごとにログを出力する
    """
    model.train()  # 訓練モードに切り替え
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_margin_loss = 0.0
    num_batches = 0
    overall_correct = 0
    overall_samples = 0

    # 各損失項の重み（ハイパーパラメータ）
    value_loss_coefficient = 0.05
    margin_loss_coefficient = 0.0001

    print_interval = 500  # ログ出力間隔（バッチ数）
    accumulated_accuracy = 0.0
    group_batches = 0

    # エポック開始時の時刻を記録
    last_checkpoint_time = time.time()

    # train_loader 内の各バッチに対してループ
    for boards, target_policies, target_values, target_margins in tqdm(train_loader, desc="Training", bar_format=bar_fmt):
        # バッチごとにデバイスへデータ移動
        boards = boards.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        target_margins = target_margins.to(device)

        optimizer.zero_grad()  # 1バッチ前の勾配をリセット
        pred_policy, (pred_value, pred_margin) = model(boards)  # モデルの予測出力を取得

        # ポリシー損失：ターゲットの対数確率に基づく損失（負の和をとり、バッチサイズで正規化）
        policy_loss = -torch.sum(target_policies * pred_policy) / boards.size(0)
        # 値損失：平均二乗誤差を計算
        value_loss = F.mse_loss(pred_value.view(-1), target_values.view(-1))
        # マージン損失：平均二乗誤差を計算
        margin_loss = F.mse_loss(pred_margin.view(-1), target_margins.view(-1))
        # 総損失は、各損失に重みをかけた和として計算
        loss = policy_loss + value_loss_coefficient * value_loss + margin_loss_coefficient * margin_loss

        loss.backward()     # 逆伝播による勾配計算
        optimizer.step()    # パラメータ更新

        if USE_TPU:
            # TPUの場合、明示的にステップをマークする（計算グラフの同期）
            xm.mark_step()

        # ロスおよび損失項の累積値を加算
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_margin_loss += margin_loss.item()
        num_batches += 1

        # バッチごとに正解率を計算（予測ラベルとターゲットのラベルが一致する割合）
        batch_pred = pred_policy.argmax(dim=1)
        batch_target = target_policies.argmax(dim=1)
        batch_accuracy = (batch_pred == batch_target).float().mean().item()
        overall_correct += (batch_pred == batch_target).sum().item()
        overall_samples += boards.size(0)
        accumulated_accuracy += batch_accuracy
        group_batches += 1

        # print_interval ごとにログ出力
        if num_batches % print_interval == 0:
            avg_accuracy = accumulated_accuracy / group_batches
            start_batch = num_batches - group_batches + 1
            end_batch = num_batches
            print(f" {start_batch:5d}～{end_batch:5d} policy accuracy : {avg_accuracy:6.4f}")
            accumulated_accuracy = 0.0
            group_batches = 0

        # 現在時刻との差がcheckpoint_interval秒を超えていればチェックポイントを保存
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval:
            # ここでは、epoch、損失等の情報をチェックポイントとして保存する
            # 各種パラメータは必要に応じて実環境の値に置き換える
            dummy_best_val_loss = 0.0
            epochs_no_improve = 0.0
            best_policy_accuracy = best_policy_accuracy  # この時点でbest_policy_accuracy
            
            save_checkpoint_nolog(model, optimizer, scheduler, epoch, dummy_best_val_loss, epochs_no_improve, best_policy_accuracy, CHECKPOINT_FILE, device)
            print(f" checkpoint at epoch {epoch}...")
            
            # タイマーをリセット
            last_checkpoint_time = current_time

    if group_batches > 0:
        avg_accuracy = accumulated_accuracy / group_batches
        print(f"Other ({group_batches} batch) policy accuracy : {avg_accuracy:6.4f}")

    if overall_samples > 0:
        overall_accuracy = overall_correct / overall_samples
        print(f"Overall policy accuracy of the latest model state in this training loop: {overall_accuracy:6.4f}")
    else:
        overall_accuracy = 0.0

    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = value_loss_coefficient * total_value_loss / num_batches
    avg_margin_loss = margin_loss_coefficient * total_margin_loss / num_batches

    # エポックごとの損失や正解率などをログ出力
    train_logger.info(f"Training iteration  total  loss: {avg_loss:.5f}")
    train_logger.info(f"Training iteration  policy loss: {avg_policy_loss:.5f}")
    train_logger.info(f"Training iteration  value  loss: {avg_value_loss:.5f}")
    train_logger.info(f"Training iteration  margin loss: {avg_margin_loss:.5f}")
    train_logger.info(f"Training iteration  policy  acc: {overall_accuracy:.5f}")

    return avg_loss

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
def _mp_fn(rank):
    """
    TPU分散環境（もしくはシングルプロセス）で実行されるメインの関数
    ・デバイスの設定、データセットの生成、モデルの構築、チェックポイント復元、
      学習ループおよび評価を実行する
    """
    # チェックポイントの自動保存間隔（ここでは1800秒=30分）
    checkpoint_interval = 1800

    # プログラム起動後の最初のループでのみresume_flagを利用して進捗チェックポイントから再開する
    # ここでは最初のループ以降 resume_flag を True にして使用する
    resume_flag = True

    if USE_TPU:
        # TPU利用環境なら、分散処理プロセスグループの初期化を確認の上、TPUデバイスを取得
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method='xla://')
        device = xm.xla_device()
        train_logger.info("Running on TPU device: {}".format(device))
    else:
        # TPU以外の場合は、GPUがあるならCUDA、なければCPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_logger.info("Running on device: {}".format(device))

    # テスト用データセットのpickleファイルのパス設定
    test_dataset_pickle = os.path.join(VAL_SGF_DIR, "test_dataset.pkl")

    # テスト用のSGFサンプルを生成またはロード
    test_samples = prepare_test_dataset(VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, True, test_dataset_pickle)
    test_dataset = AlphaZeroSGFDatasetPreloaded(test_samples)

    # DataLoader を用いてバッチ処理する
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ----- 学習ループ開始 -----
    while True:
        # 各ループで新たにネットワークのインスタンスを生成し、デバイスに配置する
        model = EnhancedResNetPolicyValueNetwork(
            board_size=BOARD_SIZE,
            num_channels=model_channels,
            num_residual_blocks=num_residual_blocks,
            in_channels=NUM_CHANNELS
        ).to(device)
        train_logger.info("model instance to {}".format(device))

        # optimizerの生成
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # schedulerの生成。 'max'（accuracyは高いほうが良い）に指定
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=factor)

        # チェックポイントから復元
        epoch, best_policy_accuracy = load_checkpoint(model, optimizer, scheduler, CHECKPOINT_FILE, device)
        train_logger.info("Initial best_policy_accuracy: {:.5f}".format(best_policy_accuracy))
        current_lr = optimizer.param_groups[0]['lr']
        
        # 直接 scheduler の属性から設定内容を取得（例として、scheduler.mode, scheduler.patience, scheduler.factor を使用）
        train_logger.info("Scheduler settings: mode: {}, patience: {:2d}, factor: {:.2f}".format(
            scheduler.mode, scheduler.patience, scheduler.factor))
        train_logger.info("Current learning rate : {:.8f}".format(current_lr))

        # エポック毎に新しい学習データセット（および DataLoader ）を生成
        training_dataset = load_training_dataset(TRAIN_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, resume_flag,  augment_all=True, max_files=number_max_files)

        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        # 1エポック分の訓練を実施
        train_one_iteration(model, train_loader, optimizer, scheduler, device, epoch, checkpoint_interval, best_policy_accuracy)
        epoch += 1

        # 訓練後、テスト用データセットでモデル評価を行いpolicy accuracyを算出
        policy_accuracy = validate_model(model, test_loader, device)

        if policy_accuracy > best_policy_accuracy:
            # 改善があった場合、最良モデルとして保存
            best_policy_accuracy = save_best_model(model, policy_accuracy, device, best_policy_accuracy)
        else:
            # 改善が見られなくても一時的な推論用モデルを保存
            save_inference_model(model, device, "inference2_model_tmp.pt")

        lr_before = optimizer.param_groups[0]['lr']
        train_logger.info("Epoch {} - Before scheduler.step(): lr = {:.8f}".format(epoch, lr_before))
        # scheduler.step() を policy_accuracy を用いて学習率の更新を行う
        scheduler.step(policy_accuracy)
        lr_after = optimizer.param_groups[0]['lr']
        train_logger.info("Epoch {} - After  scheduler.step(): lr = {:.8f}".format(epoch, lr_after))

        # ダミー値（本実装では使用していないが、チェックポイントのパラメータとして必要なもの）
        dummy_best_val_loss = 0.0
        dummy_epochs_no_improve = 0

        # 各エポック終了時にチェックポイントを保存
        save_checkpoint(model, optimizer, scheduler, epoch, dummy_best_val_loss, dummy_epochs_no_improve, best_policy_accuracy, CHECKPOINT_FILE, device)
        train_logger.info("Iteration completed. Restarting next iteration...\n")

# グローバル変数として強制再読み込みフラグを定義（プログラム起動時のみ True にする）
FORCE_RELOAD = False
# ==============================
# main処理
# ==============================
if __name__ == "__main__":

    train_logger.info("=== Starting Training and Validation Loop ===")

    if USE_TPU:
        # TPU利用時は、torch_xla.distributed.xla_multiprocessingを用いてプロセスを起動（ここではnprocs=1）
        import torch_xla.distributed.xla_multiprocessing as xmp
        nprocs = 1
        xmp.spawn(_mp_fn, args=(), nprocs=nprocs)
    else:
        # TPUでない場合、シングルプロセスで _mp_fn を呼び出す
        _mp_fn(0)
