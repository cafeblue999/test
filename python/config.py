# ------------------------------
# 必要なライブラリのインポート
# ------------------------------
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm             # 進捗表示用ライブラリ（ループの状況を視覚的に表示）
import time                         # 時間計測・待機処理に利用
from datetime import datetime, timedelta, timezone
import logging

## コマンドライン引数で PREFIX を設定（config.py で利用）
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--prefix", type=str, help="SGFディレクトリ／ファイル名のPREFIX")
parser.add_argument(
    "--force_reload",
    type=str,
    choices=["True", "False"],
    default="True",
    help="初回サイクル時に全ファイル再読み込みを行うか ('True' or 'False')"
)
args, _ = parser.parse_known_args()
if args.prefix:
    os.environ["PREFIX"] = args.prefix
force_reload_flag = (args.force_reload == "True")

# ===== 固定定義：環境切り替え用フラグ =====
# TPUやGoogle Colab環境で動作させるかどうかのフラグ。環境に応じた挙動を切り替える
# TPU/GPU 切り替えフラグ (環境変数 USE_TPU="0"/"false"/"no" で False、未指定 or "1"/"true"/"yes" で True)
USE_TPU = os.environ.get("USE_TPU", "1").lower() in ("1", "true", "yes")
USE_COLAB = True    # Google Colab 環境で実行する場合は True にする

# グローバル変数として強制再読み込みフラグを定義（プログラム起動時のみ True にする）
FORCE_RELOAD = force_reload_flag

# PREFIX は環境変数から取得（デフォルトは "4"）
PREFIX = os.environ.get("PREFIX", "4")

# プロジェクトルート（この config.py のあるフォルダの一つ上の階層）を自動取得
BASE_DIR = Path(__file__).parent.parent.resolve()

# 学習用／評価用ディレクトリ、テストZIP
TRAIN_SGF_DIR     = BASE_DIR / f"train_sgf_{PREFIX}"
TRAIN_SGFS_ZIP    = TRAIN_SGF_DIR / "train.zip"
VAL_SGF_DIR       = BASE_DIR / "test"
TEST_SGFS_ZIP     = VAL_SGF_DIR / "test_sgfs.zip"
TEST_DATASET_PKL  = VAL_SGF_DIR / "test_dataset.pkl"

# モデル出力用ディレクトリ
MODEL_OUTPUT_DIR = BASE_DIR / "models"

# ディレクトリがなければ作成（存在すればスキップ）
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# sgfファイル用進捗チェックポイントファイルのパス
INFERENCE_MODEL_PREFIX = F"inference_{PREFIX}"
CHECKPOINT_FILE_PREFIX = f"checkpoint_{PREFIX}" 
CHECKPOINT_FILE = os.path.join(BASE_DIR, CHECKPOINT_FILE_PREFIX + ".pt")

# インポート時にディレクトリを作成（親ディレクトリを含めて、存在すればスキップ）
Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ==============================
# tqdm の表示
# ==============================
tqdm_kwargs = dict(file=sys.stdout, leave=True)
# カスタムフォーマッタを定義（常に hh:mm:ss 形式で出力）
def fixed_format_interval(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

# tqdm の format_interval を上書き
tqdm.format_interval = fixed_format_interval

# プログレスバーのフォーマットを指定（tqdmで利用）
#bar_fmt = "{l_bar}{bar}| {n:>6d}/{total:>6d} [{elapsed}<{remaining}, {rate_fmt}]"
bar_fmt="{desc}:{percentage:3.0f}% {n:>4d}/{total:>4d} [{elapsed}<{remaining}, {rate_fmt}]"

# ==============================
# FileLogger（シングルトン対応版）
# ==============================
# タイムゾーン設定（日本時間）
JST = timezone(timedelta(hours=9), 'JST')

# ログ出力ディレクトリ（Colab の Drive マウント or ローカル logs フォルダ）
colab_log = Path("/content/drive/My Drive/sgf/logs")
if colab_log.exists():
    LOG_DIR = colab_log
else:
    LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# タイムスタンプは実行ごとに固定化（環境変数に保存）
if "TRAIN_LOG_TIMESTAMP" not in os.environ:
    os.environ["TRAIN_LOG_TIMESTAMP"] = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
log_timestamp = os.environ["TRAIN_LOG_TIMESTAMP"]

# ログファイルパスの決定（rank0 のみファイル出力、それ以外は捨てる）
LOG_FILE_PATH = os.path.join(LOG_DIR, f"train_log_{log_timestamp}.log")

# ログレベル（環境変数 LOG_LEVEL から制御。未指定なら INFO）
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")  # 例: "DEBUG", "INFO", "WARNING", "ERROR"

# lossログ用ディレクトリ
LOSS_LOG_DIR = LOG_DIR / "loss_logs"
LOSS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ファイル処理カウントの保存先
COUNTS_FILE = LOG_DIR / "file_process_counts.pkl"

# ==============================
# FileLogger クラス定義
# ==============================
class FileLogger:
    """ ファイルへのログ出力を管理するシンプルなロガークラス（print併用） """
    def __init__(self):
        self.log_file = open(LOG_FILE_PATH, "a", encoding="utf-8")  # ファイル追記モード
        self._level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        self.min_level = self._level_map.get(LOG_LEVEL, 20)  # 出力レベル下限

    def _write(self, level, message):
        """ レベル判定付きのファイル/コンソール出力 """
        if self._level_map.get(level, 0) < self.min_level:
            return

        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} {level}:{message}"
        print(log_line)  # コンソール出力（rank0以外も見える）
        #tqdm.write(log_line)
        self.log_file.write(log_line + "\n")
        self.log_file.flush()

    # 各ログレベルに対応するメソッド
    def debug(self, message, *args, **kwargs): self._write("DEBUG", message)
    def info(self, message, *args, **kwargs): self._write("INFO", message)
    def warning(self, message, *args, **kwargs): self._write("WARNING", message)
    def error(self, message, *args, **kwargs): self._write("ERROR", message)

    def __del__(self):
        """ ファイルクローズ処理（明示的破棄 or スコープ終了時） """
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# ==============================
# シングルトンとして取得する関数
# ==============================
# グローバルに 1 インスタンスだけ保持
_logger_instance = None

def get_logger():
    """
    シングルトンの FileLogger を返す。
    import 時に初期化されず、呼び出し時に1度だけ作成される。
    他のファイルから `from config import get_logger` で取得可能。
    """
    global _logger_instance
    if _logger_instance is None:
   #     _logger_instance = FileLogger()
   # return _logger_instance
        # Python 標準 logging.Logger を生成・設定
        logger = logging.getLogger("")
        # 既存のハンドラをクリアし、親ロガーへの伝播も無効化（重複出力防止）
        logger.handlers.clear()
        logger.propagate = False

        # 環境変数から取得した LOG_LEVEL（文字列）を数値レベルに変換
        level = getattr(logging, LOG_LEVEL, logging.INFO)
        logger.setLevel(level)

        # --- ファイルへの出力ハンドラ ---
        fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        fh.setLevel(level)
        fmt = '%(asctime)s [%(levelname)s] %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        formatter.converter = lambda ts: datetime.fromtimestamp(ts, JST).timetuple()
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # --- コンソール出力用ハンドラ（tqdm.write をやめて通常の StreamHandler を使う） ---
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        _logger_instance = logger

    return _logger_instance
