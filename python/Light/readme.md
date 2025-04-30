# プロジェクト：クラス／関数一覧

以下、各ファイルごとに定義されている**クラス**および**関数**を漏れなく列挙し、簡単な説明を付しています。

---

## config.py

### クラス

- **FileLogger**  
  ファイル＋コンソール出力対応のシンプルなロガークラス
  - `debug(msg)`, `info(msg)`, `warning(msg)`, `error(msg)` メソッド

### 関数

- **get_logger(name: str, level: int=logging.INFO)**  
  標準の `logging` モジュールに `StreamHandler` + `FileHandler` を追加したロガーを返す
- **fixed_format_interval(seconds: float) → str**  
  tqdm 用の時間表示フォーマットを `hh:mm:ss` に固定する
- **（その他 CLI 引数パース／環境変数設定／定数定義）**  
  `PREFIX`, `FORCE_RELOAD`, `USE_TPU`, `TRAIN_SGF_DIR` などのグローバル設定

---

## utils.py

### 定数

- **BOARD_SIZE**（int）
- **HISTORY_LENGTH**（int）
- **NUM_CHANNELS**（int）
- **model_channels**（int）／**num_residual_blocks**（int）
- **batch_size**, **learning_rate**, **patience**, **factor**, **number_max_files**
- **bar_fmt**（str）  
  — いずれもモデル・学習ループの設定用パラメータ

> ※ 関数は定義されていません。

---

## model.py

### クラス

- **ResidualBlock(nn.Module)**  
  ResNet の 1 段分残差ブロック
- **SelfAttention(nn.Module)**  
  空間的自己注意機構を実装する層
- **EnhancedResNetPolicyValueNetwork(nn.Module)**
  - `__init__(...)`  
    入力畳み込み／BatchNorm／残差ブロック群／ポリシーヘッド／バリューヘッドを定義
  - `forward(x) → (policy_logits, (value, margin))`  
    順伝播結果として対数確率ポリシーと勝率・マージンを返す

---

## dataset.py

### クラス

- **AlphaZeroSGFDatasetPreloaded(Dataset)**
  - `__init__(samples: List[Tuple])`
  - `__len__() → int`
  - `__getitem__(idx) → (board_tensor, policy, value, margin)`  
    事前生成済みサンプルを PyTorch Dataset として扱う

### 関数

- **parse_sgf(sgf_text: str) → Dict**  
  SGF 文字列をノード辞書リスト (`{"root":…, "nodes": […]}`) に分解
- **build_input_from_history(history: List[np.ndarray], current_player, board_size, history_length) → np.ndarray**  
  17 チャネル入力テンソルを構築
- **process_sgf_to_samples_from_text(sgf_src: str, board_size, history_length, augment_all: bool) → List[Tuple]**  
  各ノードごとに `(board, policy, value, margin)` サンプルを生成
- **prepare_test_dataset(sgf_dir, board_size, history_length, augment_all, output_file) → List[Tuple]**  
  ZIP/SGF からテスト用サンプルを生成・pickle キャッシュ
- **save_dataset(samples: List[Tuple], output_file: str) → None**  
  サンプルリストを pickle 形式で保存
- **load_dataset(output_file: str) → List[Tuple]**  
  pickle からサンプルリストを読み込み
- **validate_model(model, test_loader, device) → (policy_acc, value_mse, margin_mse, total_loss)**  
  検証データに対する精度・MSE・合計損失を計算
- **save_inference_model(model, device, model_name: str) → None**  
  TorchScript 推論モデルを生成・保存
- **save_best_acc_model(model, policy_accuracy: float, device) → None**  
  policy accuracy ベースで最良モデル保存＋古いファイル削除
- **save_best_loss_model(model, total_loss: float, device) → None**  
  total loss ベースで最良モデル保存＋古いファイル削除
- **save_checkpoint(model, optimizer, scheduler, best_total_loss, best_policy_accuracy, checkpoint_file: str) → None**  
  学習中間チェックポイントを保存
- **save_checkpoint_nolog(...) → None**  
  ログ出力なし版チェックポイント保存
- **recursive_to(data, device) → Any**  
  ネスト構造中の Tensor を再帰的に `.to(device)`
- **load_checkpoint(model, optimizer, scheduler, checkpoint_file: str, device) → (best_total_loss, best_policy_accuracy)**  
  チェックポイントから状態を復元

---

## train.py

### 関数

- **train_one_iteration(model, train_loader, optimizer, device, local_loop_cnt, rank) → float**  
  1 エポック分の訓練（順伝播／損失計算／逆伝播／更新）を実施し平均 loss を返す
- **\_mp_fn(rank: int) → None**  
  TPU/XLA 分散環境下で各プロセスが実行するメイン学習ループ
- **main() → None**  
  CLI から `_mp_fn` を呼び出すエントリポイント

---
