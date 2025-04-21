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
#bar_fmt = "{l_bar}{bar}| {n:>6d}/{total:>6d} [{elapsed}<{remaining}, {rate_fmt}]"
bar_fmt="{desc}:{percentage:3.0f}% {n:>4d}/{total:>4d} [{elapsed}<{remaining}, {rate_fmt}]"

# ==============================
# FileLogger クラス（ログ出力用）
# ==============================
from datetime import datetime, timedelta, timezone

# JST（日本標準時）
JST = timezone(timedelta(hours=9), 'JST')
# 1) ログ専用ディレクトリを定義（必要であれば先に os.makedirs で作成）
LOG_DIR = "/content/drive/My Drive/sgf/logs"
# 2) rank 環境変数取得
rank_env = os.environ.get("XRT_SHARD_ORDINAL", os.environ.get("LOCAL_RANK", "0"))
# ログファイル名をタイムスタンプ付きで作成
log_timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
# TPU の各プロセスには XRT_SHARD_ORDINAL が設定されています
rank_env = os.environ.get("XRT_SHARD_ORDINAL",
                          os.environ.get("LOCAL_RANK", "0"))
# rank0判定フラグ
is_rank0 = (rank_env == "0")
# rank0のみファイル出力。その他は/dev/null（Windowsなら'nul'）へ捨てる
if is_rank0:
    LOG_FILE_PATH = os.path.join(
        LOG_DIR,
        f"train_log_{log_timestamp}_rank{rank_env}.log"
    )
else:
    LOG_FILE_PATH = os.devnull

# 環境変数 LOG_LEVEL から出力最小レベルを取得。未定義なら INFO とする
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

class FileLogger:
    def __init__(self):
        # ログファイルを開く
        self.log_file = open(LOG_FILE_PATH, "a", encoding="utf-8")
        # ログレベルマップ
        self._level_map = {"DEBUG":10, "INFO":20, "WARNING":30, "ERROR":40}
        # 出力最小レベル（環境変数 LOG_LEVEL か WARNING）
        self.min_level = self._level_map.get(LOG_LEVEL, 20)

        # 外部からセットされるランク番号用のプレースホルダ
        self.rank = None

    def _write(self, level, message):
        # レベルフィルタ
        if self._level_map.get(level, 0) < self.min_level:
            return

        # タイムスタンプ
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        # 実行時にセットされた self.rank を優先
        # rank_str = str(self.rank) if self.rank is not None else \
        #           os.environ.get("XRT_SHARD_ORDINAL", os.environ.get("LOCAL_RANK", "0"))
        #prefix = f"[rank {rank_str}] "

        # 最終的なログ行
        log_line = f"{timestamp} {level}: {message}"

        # 出力
        print(log_line)
        self.log_file.write(log_line + "\n")
        self.log_file.flush()

    def debug(self, message, *args, **kwargs):
        self._write("DEBUG", message)

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