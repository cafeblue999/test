import os
import configparser
from config import BASE_DIR, get_logger, PREFIX

train_logger = get_logger()

# ==============================
# utils関数
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
    config.read(config_path, encoding='utf-8')            # ファイルの読み込み

    try:
        # BOARDセクション：盤のサイズ（例：19x19）を取得。指定がなければ"19"を使用
        BOARD_SIZE = int(config.get("BOARD", "BOARD_SIZE", fallback="19"))
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
        train_batch_size = int(config.get("TRAIN", "train_batch_size", fallback="256"))
        test_batch_size = int(config.get("TRAIN", "test_batch_size", fallback="256"))
        argument = int(config.get("TRAIN", "argument", fallback="0"))
        learning_rate = float(config.get("TRAIN", "learning_rate", fallback="0.001"))
        patience = int(config.get("TRAIN", "patience", fallback="10"))
        factor = float(config.get("TRAIN", "factor", fallback="0.8"))
        # 一回のデータセット生成時に処理するファイル数
        number_max_files = int(config.get("TRAIN", "number_max_files", fallback="256"))
        number_proc_files = int(config.get("TRAIN", "number_proc_files", fallback="64"))
        val_interval = int(config.get("TRAIN", "val_interval", fallback="1"))
        w_policy_loss = float(config.get("TRAIN", "w_policy_loss", fallback="1.0"))
        w_value_loss = float(config.get("TRAIN", "w_value_loss", fallback="0.1"))
        w_margin_loss = float(config.get("TRAIN", "w_margin_loss", fallback="0.0001"))
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
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "argument": argument,
        "learning_rate": learning_rate,
        "patience": patience,
        "factor": factor,
        "number_max_files": number_max_files,
        "number_proc_files": number_proc_files,
        "val_interval": val_interval,
        "w_policy_loss": w_policy_loss,
        "w_value_loss": w_value_loss,
        "w_margin_loss": w_margin_loss
    }

# 設定ファイルのパス（BASE_DIR配下にあると仮定）
CONFIG_PATH = os.path.join(BASE_DIR, f"config_py_{PREFIX}.ini")
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
train_batch_size = config_params["train_batch_size"]
test_batch_size = config_params["test_batch_size"]
argument = config_params["argument"]
learning_rate = config_params["learning_rate"]
patience = config_params["patience"]
factor = config_params["factor"]
number_max_files = config_params["number_max_files"]
number_proc_files = config_params["number_proc_files"]
val_interval = config_params["val_interval"]
w_policy_loss = config_params["w_policy_loss"]
w_value_loss = config_params["w_value_loss"]
w_margin_loss = config_params["w_margin_loss"]
