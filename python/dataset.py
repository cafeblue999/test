import os
import re
import time
import datetime
import pickle
import zipfile
import pytz
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from config import USE_TPU, USE_COLAB, get_logger, PREFIX, MODEL_OUTPUT_DIR, INFERENCE_MODEL_PREFIX, bar_fmt, TEST_SGFS_ZIP, tqdm_kwargs, LOSS_LOG_DIR, JST, COUNTS_FILE
from utils  import BOARD_SIZE, NUM_CHANNELS, w_policy_loss, w_value_loss

if USE_TPU:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader


train_logger = get_logger()

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

    # TPU 前提では入力は float32 のまま保持し、Tensor 化時に bfloat16 に変換
    return np.stack(channels, axis=0).astype(np.float32)

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
    def __init__(self, samples, file_list=None) :
        self.samples = samples
        # file_list が与えられなかった場合はダミーの None リストをセット
        if file_list is None:
            self.file_list = [None] * len(samples)
        else:
            self.file_list = file_list

        # ── ファイル処理カウンタの読み込み or 初期化 ──
        if os.path.exists(COUNTS_FILE):
            with open(COUNTS_FILE, "rb") as f:
                saved = pickle.load(f)
            # 現在の file_list のみに絞ってカウントを復元 or 0 初期化
            self.file_process_counts = {
                fp: saved.get(fp, 0)
                for fp in self.file_list
            }
        else:
            self.file_process_counts = {fp: 0 for fp in self.file_list}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        訓練用サンプル（5要素）とテスト用サンプル（4要素）の両方を扱います。
        常に3要素の Tensor を返します。
        """
        item = self.samples[idx]
        if len(item) == 4:
            inp, pol, val, _ = item
        else:
            inp, pol, val = item

        # board_tensor の生成：numpy.ndarray または Tensor を受け付け、torch.bfloat16 の Tensor に変換
        if torch.is_tensor(inp):
            board_tensor = inp.to(torch.bfloat16)
        elif isinstance(inp, np.ndarray):
            board_tensor = (
                torch.from_numpy(inp)
                .view(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
                .to(torch.bfloat16)
            )
        else:
            raise TypeError(f"Unsupported input type for dataset: {type(inp)}")

        # policy_target をラベル用の LongTensor に変換(cross_entropy に渡すラベルは本来 long 型である)
        if torch.is_tensor(pol):
            policy_tensor = pol.to(torch.long)
        else:
            policy_tensor = torch.tensor(pol, dtype=torch.long)

        # TPU: bfloat16 にキャスト
        if torch.is_tensor(val):
            value_tensor = val.to(torch.bfloat16)
        else:
            value_tensor = torch.tensor(val, dtype=torch.bfloat16)

        return board_tensor, policy_tensor, value_tensor
    
    def save_file_counts(self):
        """ファイル処理回数をディスクに保存"""
        os.makedirs(os.path.dirname(COUNTS_FILE), exist_ok=True)
        with open(COUNTS_FILE, "wb") as f:
            pickle.dump(self.file_process_counts, f)

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
        train_logger.info(f"Test dataset pickle {output_file} already exists. Loading it directly...")
        return load_dataset(output_file)

    if not os.path.exists(TEST_SGFS_ZIP):
        train_logger.info(f"Creating zip archive {TEST_SGFS_ZIP} from SGF files in {sgf_dir} ...")
        sgf_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir)
                     if f.endswith('.sgf') and "analyzed" not in f.lower()]
        with zipfile.ZipFile(TEST_SGFS_ZIP, 'w') as zf:
            for filepath in sgf_files:
                zf.write(filepath, arcname=os.path.basename(filepath))
        train_logger.info(f"Zip archive created: {TEST_SGFS_ZIP}")
    else:
        train_logger.info(f"Zip archive {TEST_SGFS_ZIP} already exists. Loading from it...")

    all_samples = []

    with zipfile.ZipFile(TEST_SGFS_ZIP, 'r') as zf:
        # SGFファイルの名前リストを取得
        sgf_names = [name for name in zf.namelist() if name.endswith('.sgf') and "analyzed" not in name.lower()]
        sgf_names.sort()  # 名前順にソート
        train_logger.info(f"TEST: Total SGF files in zip to process: {len(sgf_names)}")
        for name in tqdm(sgf_names, desc="Processing TEST SGF files"):
            try:
                sgf_src = zf.read(name).decode('utf-8')
                file_samples = process_sgf_to_samples_from_text(sgf_src, board_size, history_length, augment_all=False)
                all_samples.extend(file_samples)
            except Exception as e:
                train_logger.error(f"Error processing {name} from zip: {e}")

    # 生成されたサンプルをpickleファイルに保存
    save_dataset(all_samples, output_file)
    train_logger.info(f"TEST: Saved test dataset (total samples: {len(all_samples)}) to {output_file}")

    return all_samples

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
        train_logger.error(f"Error processing SGF text: {e}")
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
                train_logger.warning(f"Error parsing move in SGF text: {e}")
                target_policy = np.zeros(sz * sz + 1, dtype=np.float32)
                target_policy[sz * sz] = 1.0

        # データ拡張処理（dihedral変換）：augment_allがTrueなら8通り、Falseなら元のSGFサンプルのみを返す
        transforms = range(8) if augment_all else [0]
        for t in transforms:
            inp = apply_dihedral_transform(input_tensor, t)
            pol = transform_policy(target_policy, t, sz)
            samples.append((
                inp.flatten(),  # 盤面入力を1次元にフラット化
                pol,            # 変換後のターゲットポリシー
                np.array([target_value], dtype=np.float32),   # ターゲット値
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
                train_logger.warning(f"Error updating board from SGF text: {e}")
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
    train_logger.debug(f"Saved dataset to {output_file}")

def load_dataset(output_file):
    """
    pickleファイルからサンプル群を読み込む関数
    """
    with open(output_file, "rb") as f:
        samples = pickle.load(f)
    train_logger.debug(f"Loaded dataset from {output_file}")
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
    model_cpu.eval()

    # 推論のトレース用ダミー入力（形状は [1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE]）
    dummy_input = torch.randn(1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    traced_module = torch.jit.trace(model, dummy_input)  # TorchScript形式に変換
    inference_model_file = os.path.join(MODEL_OUTPUT_DIR, model_name)

    torch.jit.save(traced_module, inference_model_file)  # ファイルに保存

    train_logger.info(f"Inference model saved as {inference_model_file}")
    
    model_cpu.train()

    model_cpu.to(device)  # 元のデバイスに戻す

# ==============================
# 最良モデル保存用関数(Policy accurach)
# ==============================
def save_best_acc_model(model, policy_accuracy, device):
    """
    現在の policy_accuracy が最高値を更新したら：
      1) state_dict モデルを保存
      2) 推論モデルを保存
      3) 古いスコア付きファイルを一掃
      4) 最新の最高値を返す
    """
    # 1) 新規ファイルパスを定義
    new_model_file     = os.path.join(
        MODEL_OUTPUT_DIR, f"model_{PREFIX}_acc_{policy_accuracy:.5f}.pt"
    )
    new_inference_file = os.path.join(
        MODEL_OUTPUT_DIR, f"{INFERENCE_MODEL_PREFIX}_acc_{policy_accuracy:.5f}.pt"
    )

    # 2) モデル保存
    torch.save(model.state_dict(), new_model_file)
    train_logger.info(f"● New best acc model saved: {new_model_file}")

    #save_inference_model(model, device, os.path.basename(new_inference_file))

    # 3) 古いスコア付きファイルを一掃
    pattern = re.compile(r'^(?P<prefix>.+_)(?P<score>\d+\.\d+)\.pt$')

    groups = {}
    for fname in os.listdir(MODEL_OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        prefix = m.group('prefix')        # ex. "model_ALL_" or "inference_ALL_"
        score  = float(m.group('score'))  # ex. 0.34118
        groups.setdefault(prefix, []).append((fname, score))

    # accモデルのみ削除対象に限定
    for prefix, flist in groups.items():
        # “_acc_” プレフィックス以外はスキップ
        if not prefix.endswith("_acc_"):
            continue

        # 各プレフィックスで最大スコアを計算
        max_score = max(score for _, score in flist)
        for fname, score in flist:
            if score < max_score:
                path = os.path.join(MODEL_OUTPUT_DIR, fname)
                # 新たに保存した model/inference ファイルだけは残す
                if fname in (os.path.basename(new_model_file), os.path.basename(new_inference_file)):
                    continue
                try:
                    os.remove(path)
                    train_logger.info(f"Deleted old file: {fname}")
                except OSError as e:
                    train_logger.warning(f"Failed to delete {fname}: {e}")

# ==============================
# 最良モデル保存用関数(total loss)
# ==============================
def save_best_loss_model(model, total_loss, device):
    """
    現在の policy_accuracy が最高値を更新したら：
      1) state_dict モデルを保存
      2) 推論モデルを保存
      3) 古いスコア付きファイルを一掃
      4) 最新の最高値を返す
    """
    # 1) 新規ファイルパスを定義 
    new_model_file     = os.path.join(
        MODEL_OUTPUT_DIR, f"model_{PREFIX}_loss_{total_loss:.5f}.pt"
    )
    new_inference_file = os.path.join(
        MODEL_OUTPUT_DIR, f"{INFERENCE_MODEL_PREFIX}_loss_{total_loss:.5f}.pt"
    )

    # 2) モデル保存
    torch.save(model.state_dict(), new_model_file)
    train_logger.info(f"● New best loss model saved: {new_model_file}")

    #save_inference_model(model, device, os.path.basename(new_inference_file))

    # 3) 古いスコア付きファイルを一掃
    pattern = re.compile(r'^(?P<prefix>.+_)(?P<score>\d+\.\d+)\.pt$')

    groups = {}
    for fname in os.listdir(MODEL_OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        prefix = m.group('prefix')        # ex. "model_ALL_" or "inference_ALL_"
        score  = float(m.group('score'))  # ex. 0.34118
        groups.setdefault(prefix, []).append((fname, score))

    # 「新しく保存したファイル名」のみを絶対に残すセットを作る
    keep_fnames = {
        os.path.basename(new_model_file),
        os.path.basename(new_inference_file),
    }
    # lossモデルのみ削除対象に限定 
    for prefix, flist in groups.items():
        # “_loss_” プレフィックス以外はスキップ
        if not prefix.endswith("_loss_"):
            continue
        # 各プレフィックスで最小スコア（最良モデル）を計算
        min_score = min(score for _, score in flist)
        for fname, score in flist:
            # スコアが最小より大きい（損失が悪化している）かつ、新規モデルでなければ削除
            if score > min_score and fname not in keep_fnames:
                path = os.path.join(MODEL_OUTPUT_DIR, fname)
                try:
                    os.remove(path)
                    train_logger.info(f"Deleted old file: {fname}")
                except OSError as e:
                    train_logger.warning(f"Failed to delete {fname}: {e}")

# ==============================
# モデル検証関数（mse_criterion 廃止版）
# ==============================

def validate_model(model, test_loader, device):
    """
    テストデータを使用してモデルのポリシーとバリューを評価する関数。

    Args:
        model (torch.nn.Module): 評価対象のモデル
        test_loader (DataLoader): テスト用の DataLoader
        device (torch.device): 使用デバイス（CPU/GPU/TPU）

    Returns:
        tuple: (policy_accuracy, average_value_loss, total_loss)
    """
    model.eval()
    total_samples = 0
    policy_loss_sum = torch.zeros((), device=device)
    value_loss_sum  = torch.zeros((), device=device)
    correct_preds   = torch.zeros((), device=device)

    # デバイス対応のテストローダー
    val_loader = MpDeviceLoader(test_loader, device)

    with torch.no_grad():
        for boards, t_policies, t_values in tqdm(
            val_loader,
            desc="Validation",
            bar_format=bar_fmt,
            position=0,
            **tqdm_kwargs
        ):
            # デバイス転送
            boards      = boards.to(device, non_blocking=True, dtype=torch.bfloat16)
            t_policies  = t_policies.to(device, non_blocking=True)
            t_values    = t_values.to(device, non_blocking=True, dtype=torch.bfloat16)

            batch_size = boards.size(0)
            total_samples += batch_size

            # 推論
            p_logits, v_pred = model(boards)

            # 精度計算
            preds  = p_logits.argmax(dim=1)
            labels = t_policies.argmax(dim=1)
            correct_preds += (preds == labels).sum()

            # 損失集計
            policy_loss_sum += F.cross_entropy(p_logits, labels, reduction='sum')
            value_loss_sum  += F.mse_loss(v_pred.view(-1), t_values.view(-1), reduction='sum')

            # TPU同期
            if USE_TPU:
                xm.mark_step()

    # 平均損失と精度計算
    avg_policy_loss = (policy_loss_sum / total_samples).item()
    avg_value_loss  = (value_loss_sum  / total_samples).item()
    policy_accuracy = (correct_preds.float() / total_samples).item()
    total_loss      = w_policy_loss * avg_policy_loss + w_value_loss * avg_value_loss
    
    train_logger.info(f"Validation end: p_acc:{policy_accuracy:.5f} w_total_loss:{total_loss:.5f} p_loss:{avg_policy_loss:.5f} v_loss:{avg_value_loss:.5f}")
    
    # Validation metrics をファイル出力
    ts = datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    # policy loss
    with open(os.path.join(LOSS_LOG_DIR, "validation_policy_loss.log"), "a", encoding="utf-8") as f:
        f.write(f"{ts},{avg_policy_loss:.5f}\n")
    # value loss
    with open(os.path.join(LOSS_LOG_DIR, "validation_value_loss.log"), "a", encoding="utf-8") as f:
        f.write(f"{ts},{avg_value_loss:.5f}\n")
    # policy accuracy
    with open(os.path.join(LOSS_LOG_DIR, "validation_policy_acc.log"), "a", encoding="utf-8") as f:
        f.write(f"{ts},{policy_accuracy:.5f}\n")

    model.train()

    return policy_accuracy, avg_value_loss, total_loss

def save_checkpoint(model, optimizer, scheduler, best_total_loss, best_policy_accuracy, checkpoint_file):
    """
    トレーニング途中のモデル状態を保存する関数。

    Args:
        model (torch.nn.Module): 保存対象のモデル
        optimizer (Optimizer): モデルのオプティマイザ
        scheduler (_LRScheduler): 学習率スケジューラ
        best_total_loss (float): これまでの最小合計損失
        best_policy_accuracy (float): これまでの最高ポリシー精度
        checkpoint_file (str): 保存先ファイルパス
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_total_loss': best_total_loss,
        'best_policy_accuracy': best_policy_accuracy,
    }
    torch.save(checkpoint, checkpoint_file)
    train_logger.info(f"Checkpoint saved to {checkpoint_file}")
    
def save_checkpoint_nolog(model, optimizer, scheduler, best_total_loss, best_policy_accuracy, checkpoint_file):
    """
    訓練中のコンソール表示を行わない版チェックポイント保存
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_total_loss': best_total_loss,
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

# ==============================
# チェックポイント復元関数
# ==============================
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
               
        # トータルロスベースのベストを読み込み（存在しなければ inf）
        best_total_loss = checkpoint.get('best_total_loss', 10.0)

        # 精度ベースのベストを読み込み（存在しなければ 0.0）
        best_policy_accuracy = checkpoint.get('best_policy_accuracy', 0.0)
               
        return  best_total_loss, best_policy_accuracy  
    else:
        train_logger.info("No checkpoint found. Starting from scratch.")
        
        return  10.0, 0.0


