## コード処理内容及び仕様書（詳細版）

本書は、以下5つのファイル (`config.py`, `utils.py`, `model.py`, `dataset.py`, `train.py`) からなるプロジェクトの処理内容および仕様を、**処理順**に従いかつ**設計意図付き**で詳細に記述したドキュメントである。
印刷配布可能な形式としてMarkdownベースで構成している。

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

### 1.1 概要と設計意図
- **目的**：SGFデータ処理および学習パラメータに関する設定を、簡潔かつ一元的に管理する。
- **CLI引数対応により、実験バリエーション（接頭辞や再読み込み）に柔軟に対応可能。**

### 1.2 詳細処理
1. `argparse` を使って以下を読み取る：
   - `--prefix`：SGFデータセットの識別用接頭辞（例：3 → train_sgf_3）
   - `--force_reload`：pickle化済みデータセットを再構築するかどうか
2. `datetime.timezone(datetime.timedelta(hours=+9), 'JST')` による日本時間固定。ロギングとファイル命名の一貫性のため。
3. `get_logger()` 関数：ファイル＆コンソール両対応のロガーを統一形式で生成。視認性とデバッグ効率の向上を意図。
4. 複数のグローバル定数を定義：
   - SGFディレクトリ（TRAIN_SGF_DIR 等）
   - モデル保存ディレクトリ（MODEL_OUTPUT_DIR）
   - チェックポイントファイルの命名（CHECKPOINT_FILE）

---

<a name="utilspy"></a>
## 2. utils.py

### 2.1 概要と設計意図
- **目的**：定数定義と各所から再利用される関数を一元化。設定ミス防止、可読性向上。

### 2.2 処理内容
- `BOARD_SIZE=19`, `HISTORY_LENGTH=8`, `NUM_CHANNELS=17` などは、囲碁の盤面処理における入力特徴量定義。
- `model_channels`, `num_residual_blocks`：ResNetの幅・深さを制御。チューニング可能。
- `bar_fmt`：プログレスバーの形式を明示的に統一。Colabなどで崩れないよう配慮。

---

<a name="modelpy"></a>
## 3. model.py

### 3.1 概要と設計意図
- **目的**：AlphaZero型Policy-Valueネットワークの実装
- ResNet構造により、囲碁における局所特徴・大局的判断を両立させる

### 3.2 処理構造
1. `EnhancedResNetPolicyValueNetwork` クラスを定義。PyTorch nn.Module を継承。
2. `__init__()` にて以下を構築：
   - 初期畳み込み（Conv2D）とBatchNormで特徴抽出
   - 残差ブロックをfor文で構築（深さパラメータ指定）
   - ポリシーヘッド：出力空間は19x19（盤面）+1（パス）
   - バリューヘッド：最終的にtanhにより [-1, 1] の勝率スカラーへ
3. `forward()` にて：
   - 特徴マップ → 各ブロック通過 → 並列に2ヘッドに分岐

### 3.3 理由
- ポリシーは確率分布なのでlogits→softmax
- バリューは勝率スカラーなのでtanh（学習安定のため）
- 残差構造は学習深度を確保しつつ勾配消失を抑制

---

<a name="datasetpy"></a>
## 4. dataset.py

### 4.1 概要と設計意図
- **目的**：SGF形式の囲碁棋譜をAlphaZeroトレーニング用データに変換。
- zip対応、データ前処理、死活除去、特徴展開をすべてこのモジュール内で完結。

### 4.2 主な構成
#### 1. `AlphaZeroSGFDatasetPreloaded`
- zip内SGFも含め、SGF全体を読み取り、パース後にメモリ展開
- `__len__`, `__getitem__` を実装し、DataLoaderから使える
- 特徴量は 8履歴 + 現局面 + 着手権 + 石の存在などを17チャネルで構成

#### 2. `prepare_test_dataset`
- 検証用のSGFデータを同様に一括パースし、固定的に返す
- epoch間でバリデーションデータを再利用する構造

#### 3. `process_sgf_to_samples_from_text`
- SGF文字列を直接渡して処理（テスト・Colab向け）
- SGFパーサは内部関数で整備済み

#### 4. zip対応とキャッシュ
- `train_KK_XXXX.zip` などのzipアーカイブを直接読み込み展開
- pickle形式に変換してキャッシュ高速化（再実行時）

### 4.3 設計意図
- 巨大なSGFデータでも分散ローディング＆メモリ高速展開で学習高速化
- zip対応：Google Drive などでSGFを圧縮管理する前提設計

---

<a name="trainpy"></a>
## 5. train.py

### 5.1 概要と設計意図
- **目的**：TPU/xmp環境下での高速分散学習を自動制御
- 複数rankに同じSGFファイルが重複せずに割り当てられるよう制御
- checkpoint/バリデーション/モデル保存/シード制御すべてを統括

### 5.2 初期処理
1. 各種定数＆ロガー取得（`train_logger = get_logger()`）
2. TPU使用環境下では `xmp.spawn(_mp_fn)` によって8プロセス起動

### 5.3 `_mp_fn(rank)` 処理内容（メイン学習処理）
1. TPUプロセス情報取得：`ordinal`, `device`, `world_size`
2. SGF zipファイル一覧列挙 → `number_max_files` 単位でサンプリング
3. 学習epochループ（`while True`）開始：
   - シード設定（epoch+rankによるユニークシード）
   - モデル・オプティマイザ・ReduceLROnPlateau を初期化
   - `DistributedSampler` → `DataLoader` → `MpDeviceLoader` で分散学習ローダ構築
   - forループで各バッチを処理：
     - `forward()` → 損失（policy, value, margin）を加重和で合成
     - `backward()` → `step()` → 損失をロガーとファイルへ記録
     - `local_loop_cnt` ごとにチェックポイント＆ベストモデル保存
     - `validate_model()` により rank 0 だけ検証実施

### 5.4 設計意図
- **rankごとに異なるデータサンプリング**：処理の重複を避け効率化
- **policy accuracy と total loss の両軸でベストモデル保存**：実務での汎化性能と整合
- **無限ループで自動学習**：Google Colab の割り当て時間で自動継続させる意図
- **CSVログ出力**：Google Sheet やグラフ描画を想定したフォーマット

---

以上。各ファイル間は明確な責任分離を意識しており、実験管理・学習再現性・分散効率に配慮された設計となっている。

