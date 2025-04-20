# ------------------------------
# 必要なライブラリのインポート
# ------------------------------
import os
from tqdm import tqdm             # 進捗表示用ライブラリ（ループの状況を視覚的に表示）
import torch.distributed as dist    # 分散学習に必要なモジュール
import time                         # 時間計測・待機処理に利用
from datetime import datetime, timedelta, timezone

# ===== 固定定義：環境切り替え用フラグ =====
# TPUやGoogle Colab環境で動作させるかどうかのフラグ。環境に応じた挙動を切り替える
USE_TPU = True      # TPU を利用する場合は True にする
USE_COLAB = True    # Google Colab 環境で実行する場合は True にする

#  TPU利用時に必要なtorch_xla関連モジュールをインポート
if USE_TPU:
    import torch_xla
    import torch_xla.core.xla_model as xm         # TPUデバイスの取得に使用
    import torch_xla.distributed.xla_backend       # TPU向けの分散学習の設定

# Google Colab利用時にGoogle Driveをマウントする
#if USE_COLAB:
#    try:
#        from google.colab import drive
#        # Google Drive を強制的にリマウントする
#        drive.mount('/content/drive', force_remount=True)
#    except ImportError:
#        print("Google Colab module not found.")

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
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint3.pt")  # チェックポイントファイルのパス
else:
    # ローカル環境（例: Windows環境）用のディレクトリ設定
    BASE_DIR = r"D:\igo\simple2"
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf")
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
    CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint3.pt")

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
# FileLogger クラス（ログ出力用）
# ==============================
from datetime import datetime, timedelta, timezone

# JST（日本標準時）
JST = timezone(timedelta(hours=9), 'JST')

# ログファイル名をタイムスタンプ付きで作成
log_timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, f"train_log_{log_timestamp}.log")

class FileLogger:
    def __init__(self):
        self.log_file = open(LOG_FILE_PATH, "a", encoding="utf-8")

    def _write(self, level, message):
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} {level}: {message}"
        print(log_line)  # 標準出力
        self.log_file.write(log_line + "\n")  # ファイル出力
        self.log_file.flush()

    def info(self, message, *args, **kwargs):
        self._write("INFO", message)

    def warning(self, message, *args, **kwargs):
        self._write("WARNING", message)

    def error(self, message, *args, **kwargs):
        self._write("ERROR", message)

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# ロガーインスタンス生成
sgf_logger = FileLogger()
train_logger = FileLogger()

# sgfファイル用進捗チェックポイントファイルのパス
PROGRESS_CHECKPOINT_FILE = os.path.join(BASE_DIR, "progress_checkpoint3.pkl")