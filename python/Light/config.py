# ------------------------------
# 必要なライブラリのインポート
# ------------------------------
import os
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
USE_TPU = True      # TPU を利用する場合は True にする
USE_COLAB = True    # Google Colab 環境で実行する場合は True にする

# グローバル変数として強制再読み込みフラグを定義（プログラム起動時のみ True にする）
FORCE_RELOAD = force_reload_flag

# ==============================
# ディレクトリ設定
# ==============================
# ディレクトリ設定（PREFIX は環境変数から取得）
PREFIX = "ALL"
PREFIX = os.environ.get("PREFIX", PREFIX)

if USE_COLAB:
    # Google Colab環境向けのディレクトリ設定。Google Drive内の特定のフォルダを利用する
    BASE_DIR = "/content/drive/My Drive/sgf"              # ベースディレクトリ
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, f"train_sgf_{PREFIX}")  # 学習用SGFファイルのディレクトリ
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")            # 評価用SGFファイルのディレクトリ
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")# テスト用SGFファイル群をまとめたZIPファイル
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")      # モデル出力用ディレクトリ

else:
    # ローカル環境（例: Windows環境）用のディレクトリ設定
    BASE_DIR = r"D:\igo\simple2"
    TRAIN_SGF_DIR = os.path.join(BASE_DIR, "train_sgf")
    VAL_SGF_DIR = os.path.join(BASE_DIR, "test")
    TEST_SGFS_ZIP = os.path.join(VAL_SGF_DIR, "test_sgfs.zip")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")

# sgfファイル用進捗チェックポイントファイルのパス
INFERENCE_MODEL_PREFIX = F"inference_{PREFIX}"
CHECKPOINT_FILE_PREFIX = f"checkpoint_{PREFIX}" 
CHECKPOINT_FILE = os.path.join(BASE_DIR, CHECKPOINT_FILE_PREFIX + ".pt")

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
# FileLogger（シングルトン対応版）
# ==============================
# タイムゾーン設定（日本時間） -----
JST = timezone(timedelta(hours=9), 'JST')

# ログ出力ディレクトリ（Google Drive直下） -----
LOG_DIR = "/content/drive/My Drive/sgf/logs"

# タイムスタンプは実行ごとに固定化（環境変数に保存） -----
if "TRAIN_LOG_TIMESTAMP" not in os.environ:
    os.environ["TRAIN_LOG_TIMESTAMP"] = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
log_timestamp = os.environ["TRAIN_LOG_TIMESTAMP"]

# ログファイルパスの決定（rank0 のみファイル出力、それ以外は捨てる） -----
LOG_FILE_PATH = os.path.join(LOG_DIR, f"train_log_{log_timestamp}.log")

# ログレベル（環境変数 LOG_LEVEL から制御。未指定なら INFO） -----
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")  # 例: "DEBUG", "INFO", "WARNING", "ERROR"

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
        #print(log_line)  # コンソール出力（rank0以外も見える）
        tqdm.write(log_line)
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

        # --- tqdm.write() 用ハンドラ ---
        class TqdmLoggingHandler(logging.Handler):
            def __init__(self, level=logging.NOTSET):
                super().__init__(level)
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg)
                except Exception:
                    self.handleError(record)

        th = TqdmLoggingHandler()
        th.setLevel(level)
        th.setFormatter(formatter)
        logger.addHandler(th)

        _logger_instance = logger
    return _logger_instance
