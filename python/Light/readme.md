## コード処理内容及び仕様書（詳細・関数別解説付き）

本書は、以下 5 つのファイル (`config.py`, `utils.py`, `model.py`, `dataset.py`, `train.py`) からなるプロジェクトの処理内容および仕様を、**処理順・関数単位**で詳細に記述したドキュメントである。
印刷配布可能な形式として Markdown ベースで構成している。

---

### 目次

1. [config.py](#configpy)
2. [utils.py](#utilspy)
3. [model.py](#modelpy)
4. [dataset.py](#datasetpy)
5. [train.py](#trainpy)

---

<a name="configpy"></a>

## 1. config.py

### 関数：`get_logger(name: str, level=logging.INFO)`

- **目的**：ファイルおよびコンソールにログを出力するロガーを構築
- **処理内容**：
  1. `logging.getLogger(name)` によりロガーインスタンス取得
  2. ログレベルを設定（デフォルト：INFO）
  3. `StreamHandler`（コンソール）と `FileHandler`（log/train.log）を追加
  4. フォーマット：時間付き `[INFO] メッセージ` 形式

---

<a name="utilspy"></a>

## 2. utils.py

### 定数群

- `BOARD_SIZE`：囲碁盤のサイズ（19）
- `HISTORY_LENGTH`：履歴の深さ（過去 8 手）
- `NUM_CHANNELS`：学習モデル入力チャネル数（17）
- `model_channels`, `num_residual_blocks`：ResNet 設定
- `batch_size`, `learning_rate`, `patience`, `factor`, `number_max_files`：学習パラメータ
- `bar_fmt`：tqdm のプログレスバー用書式文字列

---

<a name="modelpy"></a>

## 3. model.py

### クラス：`EnhancedResNetPolicyValueNetwork`

- **目的**：囲碁の盤面を入力とし、着手確率（policy）と勝率（value）を出力する。

#### `__init__()`

- **引数**：`board_size`, `channels`, `num_blocks`, `input_channels`
- **処理内容**：
  - 初期 Conv + BatchNorm
  - `num_blocks`回の ResBlock（`self.res_blocks`）
  - `policy_head`（盤面全体＋パス用 Softmax）
  - `value_head`（出力 1 つ、tanh）

#### `forward(x)`

- **引数**：Tensor (B, C, 19, 19)
- **出力**：
  - `policy`: Tensor (B, 362)
  - `value`: Tensor (B, 1)
  - `margin`: Tensor (B, 1)
- **内部処理**：ResNet 構造で特徴抽出 → ヘッド分岐して出力計算

---

<a name="datasetpy"></a>

## 4. dataset.py

### クラス：`AlphaZeroSGFDatasetPreloaded`

- **目的**：事前に SGF ファイルをすべて読み込み、学習用サンプルに変換する Dataset クラス

#### `__init__(self, samples)`

- `samples`：局面データ（特徴・policy・value・margin）
- 全体をリストとして保持

#### `__len__()`

- 保持するサンプル数を返す

#### `__getitem__(idx)`

- 指定されたサンプルインデックスの局面を返す
- `board`, `target_policy`, `target_value`, `target_margin`

---

### 関数：`prepare_test_dataset()`

- SGF バリデーションデータを読み込み、テスト用に整形して返す

### 関数：`load_progress_checkpoint()`／`save_progress_checkpoint()`

- pickle ファイルによる進捗保存・復元

### 関数：`process_sgf_to_samples_from_text(text)`

- SGF ファイルの文字列を直接パースして、訓練サンプル化

---

<a name="trainpy"></a>

## 5. train.py

### 関数：`validate_model(model, test_loader, device)`

- **目的**：検証データに対する policy accuracy, value loss, margin loss を計算
- **処理内容**：
  1. `model.eval()` 設定
  2. 各バッチで `forward` 実行し、出力取得：`policy, value, margin`
  3. クロスエントロピー：`F.cross_entropy(policy_pred, target_policy)`
  4. MSE：`F.mse_loss(value_pred, target_value)`, `F.mse_loss(margin_pred, target_margin)`
  5. Accuracy 計算：`argmax`を比較して正解率を累積
  6. Tensor で同期して平均を返す（TPU 並列考慮）

### 関数：`save_checkpoint(model, optimizer, scheduler, epoch, best_policy_accuracy, best_total_loss)`

- モデル構造・重み、オプティマイザ状態、スケジューラ、スコアを辞書に保存
- `torch.save()` で `checkpoint_X.pt` として出力

### 関数：`load_checkpoint(...)`

- 上記チェックポイントを読み込み、model/load_state_dict、optimizer.load_state_dict 等を適用

### 関数：`save_best_model(model, policy_accuracy, device)`

- 現在の policy accuracy が過去最高であれば、
  - `model_{prefix}_acc_{score}.pt` として保存
  - `save_inference_model()` により軽量モデルも保存

### 関数：`save_inference_model(model, device, output_file)`

- `model.to(device)` で配置後、`torch.jit.trace(model, dummy_input)` → `save()` で保存

### 関数：`_mp_fn(rank)`

- TPU/XLA 分散学習の各プロセスが実行する本体関数

#### 処理概要：

1. SGF zip ファイル読み込み → ランダムに分割サンプリング
2. `DistributedSampler` → `DataLoader` 構築（rank ごとに異なるサンプル）
3. **損失計算の詳細**：

```python
policy_loss = F.cross_entropy(policy_preds, target_policies)
value_loss  = F.mse_loss(value_preds.squeeze(1), target_values)
margin_loss = F.mse_loss(margin_preds.squeeze(1), target_margins)
```

4. **合計損失**：重み付き和

```python
total_loss = w_policy_loss * policy_loss + w_value_loss * value_loss + w_margin_loss * margin_loss
```

5. `total_loss.backward()` → `optimizer.step()` で更新
6. validation タイミングで `validate_model()` を実行
7. `best_policy_accuracy` や `best_total_loss` を更新判定して保存

---

以上が関数単位での処理内容および設計理由、損失計算など関数内部ロジックまで含めた詳細な仕様解説である。さらに必要であれば、コード抜粋や図式化なども可能。
