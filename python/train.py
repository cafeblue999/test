import os
import random
import time
import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EnhancedResNetPolicyValueNetwork

from dataset import AlphaZeroSGFDatasetPreloaded, prepare_test_dataset, process_sgf_to_samples_from_text, load_progress_checkpoint, save_progress_checkpoint, save_checkpoint_nolog, validate_model, save_best_model, save_inference_model, load_checkpoint, print_tpu_memory

from config import USE_TPU, TRAIN_SGF_DIR, VAL_SGF_DIR, CHECKPOINT_FILE, bar_fmt,sgf_logger, train_logger

from utils import BOARD_SIZE, HISTORY_LENGTH, NUM_CHANNELS, num_residual_blocks, model_channels, batch_size, learning_rate, patience, factor, number_max_files

# TPU用の dist, xm が必要なら明示的に import
import torch.distributed as dist
if USE_TPU:
    import torch_xla.core.xla_model as xm

# dataset.save_checkpoint, load_checkpoint の代わりに、train.py 内で再定義
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                    epochs_no_improve, best_policy_accuracy,
                    checkpoint_file, device, batch_idx, base_seed):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'best_policy_accuracy': best_policy_accuracy,
        'batch_idx': batch_idx,         # ← これまでのバッチ位置
        'base_seed': base_seed          # ← 追加：乱数シード
    }
    torch.save(checkpoint, checkpoint_file)
    sgf_logger.info(f"Checkpoint saved: epoch={epoch}, batch_idx={batch_idx}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_file, device):
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        epoch      = checkpoint.get('epoch', 0)
        batch_idx  = checkpoint.get('batch_idx', -1)
        best_acc   = checkpoint.get('best_policy_accuracy', 0.0)
        base_seed  = checkpoint.get('base_seed', None)  # ← 追加で取得
        return epoch, batch_idx, best_acc, base_seed
    else:
        return 0, -1, 0.0, None

# グローバル変数（未処理のSGFファイルリスト）
remaining_sgf_files = []

# グローバル変数として強制再読み込みフラグを定義（プログラム起動時のみ True にする）
FORCE_RELOAD = False

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

    # 今回処理するファイルリストを取り出す（max_files 件まで）
    if len(remaining_sgf_files) < max_files:
        files_to_process = remaining_sgf_files.copy()
    else:
        files_to_process = remaining_sgf_files[:max_files]

    all_samples = []
    # ファイルを1件ずつ処理し、処理後すぐにチェックポイントを更新
    for sgf_file in files_to_process:
        try:
            with open(sgf_file, "r", encoding="utf-8") as f:
                sgf_src = f.read()
            file_samples = process_sgf_to_samples_from_text(
                sgf_src, board_size, history_length, augment_all
            )
            all_samples.extend(file_samples)
        except Exception as e:
            sgf_logger.error(f"Error processing file {sgf_file}: {e}")

        # 処理済みファイルを remaining_sgf_files から削除
        remaining_sgf_files.remove(sgf_file)
    
    # 最後に1回だけ進捗を保存する（ログ出力もこの時1回だけ）
    save_progress_checkpoint(remaining_sgf_files)

    # サンプルをシャッフルして返却
    random.shuffle(all_samples)
    sgf_logger.info(f"Processed {len(files_to_process)} files; total samples this cycle: {len(all_samples)}")

    print_tpu_memory()

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
def train_one_iteration(model, train_loader, optimizer, scheduler, device,
                        epoch, checkpoint_interval, best_policy_accuracy,
                        resume, start_batch_idx,
                        checkpoint_file, base_seed):
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

    # バッチ単位の再開用インデックスを初期化
    last_batch_idx = -1

    # train_loader 内の各バッチに対してループ
    for i, (boards, target_policies, target_values, target_margins) in enumerate(
            tqdm(train_loader, desc="Training", bar_format=bar_fmt)):

        # resume=True のとき、start_batch_idx までのバッチをスキップ
        if resume and i <= start_batch_idx:
            continue

        # 最後に処理したバッチ番号を記録
        last_batch_idx = i

        # ── 以降は既存のバッチ処理ロジック ──
        boards = boards.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        target_margins = target_margins.to(device)

        optimizer.zero_grad()
        pred_policy, (pred_value, pred_margin) = model(boards)
        policy_loss = -torch.sum(target_policies * pred_policy) / boards.size(0)
        value_loss = F.mse_loss(pred_value.view(-1), target_values.view(-1))
        margin_loss = F.mse_loss(pred_margin.view(-1), target_margins.view(-1))
        loss = policy_loss + value_loss_coefficient * value_loss + margin_loss_coefficient * margin_loss

        loss.backward()
        optimizer.step()

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

        # 定期チェックポイント保存時に batch_idx を渡す
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval:
            save_checkpoint_nolog(
                model, optimizer, scheduler,
                epoch, 0.0, 0.0, best_policy_accuracy,
                CHECKPOINT_FILE, device,
                batch_idx=last_batch_idx
            )
            print(f" - checkpoint at epoch {epoch}...")

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

    return avg_loss, last_batch_idx

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
from concurrent.futures import ThreadPoolExecutor

def _mp_fn(rank):
    checkpoint_interval = 1800
    resume_flag = True

    # デバイス初期化
    if USE_TPU:
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method='xla://')
        device = xm.xla_device()
        train_logger.info(f"Running on TPU device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_logger.info(f"Running on device: {device}")

    # テストデータ準備
    test_dataset_pickle = os.path.join(VAL_SGF_DIR, "test_dataset.pkl")
    test_samples = prepare_test_dataset(
        VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, True, test_dataset_pickle
    )
    test_loader = DataLoader(
        AlphaZeroSGFDatasetPreloaded(test_samples),
        batch_size=batch_size, shuffle=False
    )

    executor = ThreadPoolExecutor(max_workers=1)
    future_ds = executor.submit(lambda: load_training_dataset(
        TRAIN_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH,
        resume_flag, augment_all=True, max_files=number_max_files
    ))
    resume_flag = False

    resume_epoch, resume_batch_idx = 0, -1
    mid_epoch_resumed = False
    epoch, best_policy_accuracy = 0, 0.0

    try:
        while True:
            seed = base_seed + epoch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # プリフェッチ済みデータセットを取得
            training_dataset = future_ds.result()

            # 次エポックのプリフェッチ開始（非同期）
            future_ds = executor.submit(lambda: load_training_dataset(
                TRAIN_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH,
                resume_flag, augment_all=True, max_files=number_max_files
            ))

            # モデル・最適化・スケジューラ
            model = EnhancedResNetPolicyValueNetwork(
                BOARD_SIZE, model_channels, num_residual_blocks, NUM_CHANNELS
            ).to(device)
            train_logger.info(f"model instance to {device}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=patience, factor=factor
            )

            # checkpoint 復元
            resume_epoch, best_policy_accuracy, resume_batch_idx, base_seed = load_checkpoint(
                model, optimizer, scheduler, CHECKPOINT_FILE, device
            )
            if base_seed is None:
                base_seed = random.SystemRandom().randint(0, 2**31 - 1)

            epoch = resume_epoch
            train_logger.info(
                f"Initial best_policy_accuracy: {best_policy_accuracy:.5f}, "
                f"resume at epoch={resume_epoch}, batch={resume_batch_idx}"
            )

            # シャッフル再現のため generator 構築
            generator = torch.Generator(device=device)
            generator.manual_seed(base_seed + epoch)

            train_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator
            )

            # 中断復元かどうかの判定
            do_resume = (not mid_epoch_resumed) and (epoch == resume_epoch) and (resume_batch_idx >= 0)
            start_idx = resume_batch_idx if do_resume else -1

            avg_loss, last_batch_idx = train_one_iteration(
                model, train_loader, optimizer, scheduler,
                device, epoch, checkpoint_interval, best_policy_accuracy,
                resume=do_resume, start_batch_idx=start_idx
            )
            mid_epoch_resumed = mid_epoch_resumed or do_resume

            del training_dataset, train_loader
            gc.collect()

            # 評価と保存
            policy_accuracy = validate_model(model, test_loader, device)
            if policy_accuracy > best_policy_accuracy:
                best_policy_accuracy = save_best_model(
                    model, policy_accuracy, device, best_policy_accuracy
                )
            else:
                save_inference_model(model, device, "inference3_model_tmp.pt")

            train_logger.info(
                f"Epoch {epoch} - Before scheduler.step(): "
                f"lr = {optimizer.param_groups[0]['lr']:.8f}"
            )
            scheduler.step(policy_accuracy)
            train_logger.info(
                f"Epoch {epoch} - After  scheduler.step(): "
                f"lr = {optimizer.param_groups[0]['lr']:.8f}"
            )

            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                0.0, 0, best_policy_accuracy,
                CHECKPOINT_FILE, device,
                batch_idx=-1,
                base_seed=base_seed
            )
            train_logger.info("Iteration completed. Restarting next iteration...\n")
            epoch += 1
    finally:
        executor.shutdown()


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
