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
import pickle
import torch_xla.core.xla_model as xm
import torch.distributed as dist

from dataset import (
    AlphaZeroSGFDatasetPreloaded,
    prepare_test_dataset,
    load_progress_checkpoint,
    save_progress_checkpoint,
    save_checkpoint,
    save_checkpoint_nolog,
    validate_model,
    save_best_model,
    save_inference_model,
    load_checkpoint,
    process_sgf_to_samples_from_text
)

from config import PREFIX, USE_TPU, FORCE_RELOAD, TRAIN_SGF_DIR, VAL_SGF_DIR, MODEL_OUTPUT_DIR, INFERENCE_MODEL_PREFIX, CHECKPOINT_FILE, PROGRESS_CHECKPOINT_FILE, bar_fmt, get_logger

from utils import BOARD_SIZE, HISTORY_LENGTH, NUM_CHANNELS, num_residual_blocks, model_channels, batch_size, learning_rate, patience, factor, number_max_files, CONFIG_PATH

train_logger = get_logger()

# グローバル変数（未処理のSGFファイルリスト）
remaining_sgf_files = []
# 全SGFファイル数（初回サイクル時に記録）
TOTAL_SGF_FILES = None

def prepare_train_dataset_cycle(sgf_dir, board_size, history_length, resume_flag, augment_all, max_files, rank=0, nprocs=1):
    """
    指定フォルダ内のSGFファイルから、1サイクル分の学習サンプルを生成する関数。
    ・resume_flag が True の場合、進捗チェックポイントから残りファイルリストを読み込む。
    ・ただし、プログラム起動直後は FORCE_RELOAD が True なので、常に全件再読み込みし、
      その後 FORCE_RELOAD を False にすることで、以降は前回の進捗チェックポイントを利用する。
    ・ファイル全体をランダムな順序に並べ替え、max_files 件分だけ処理する。
    ・処理後、残りファイルリストの進捗を保存する。
    """
    global remaining_sgf_files, FORCE_RELOAD, TOTAL_SGF_FILES

    # resume_flag が True かつ FORCE_RELOAD が False の場合のみ、進捗チェックポイントからロード
    if resume_flag and not FORCE_RELOAD:
        train_logger.info(f"[rank {rank}] pre_train_ds_cycle: reloading remaining files.")
        remaining = load_progress_checkpoint()
        if remaining is not None:
            remaining_sgf_files = remaining

    # FORCE_RELOAD が True または remaining_sgf_files が空の場合、全ファイルを再読み込みする
    # 初回または remaining_sgf_files が空のときは rank0 のみ再生成、その後全 rank で同期
    if FORCE_RELOAD or not remaining_sgf_files:
        train_logger.info(f"[rank {rank}] pre_train_ds_cycle: all files reading start.") 
        if rank == 0:
            all_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir)
                         if f.endswith('.sgf') and "analyzed" not in f.lower()]

            if not all_files:
                train_logger.error(f"No SGF files found in directory: {sgf_dir}")
                # 例外を投げて異常終了させる
                raise RuntimeError(f"No SGF files to process in '{sgf_dir}'")

            random.shuffle(all_files)
            remaining_sgf_files = all_files
            # 初回サイクル時に全ファイル数を記録
            TOTAL_SGF_FILES = len(all_files)
            
            train_logger.info(f"[rank {rank}] pre_train_ds_cycle:Regenerated the random order of all SGF files : {len(all_files)} (FORCE_RELOAD was {FORCE_RELOAD})")
            
            # 再読み込み後はフラグをリセット
            FORCE_RELOAD = False

        # rank0 が remaining_sgf_files を全 rank に broadcast
        if nprocs > 1:
            if USE_TPU:
                # TPU環境では xm.rendezvous でパスリストを共有
                train_logger.info(f"[rank {rank}] pre_train_ds_cycle: pickle dump start.") 
                payload = pickle.dumps(remaining_sgf_files if rank == 0 else None)
                train_logger.info(f"[rank {rank}] pre_train_ds_cycle: xm.remdevous start.") 
                payload_list = xm.rendezvous('pre_train_ds_cycle_paths', payload)
                train_logger.info(f"[rank {rank}] pre_train_ds_cycle: pickle load start.") 
                remaining_sgf_files = pickle.loads(payload_list[0])
                train_logger.info(f"[rank {rank}] pre_train_ds_cycle: broadcast via rendezvous")
                FORCE_RELOAD = False
            else:
                # 通常の分散環境では dist.broadcast_object_list を使用
                buf = [remaining_sgf_files] if rank == 0 else [None]
                dist.broadcast_object_list(buf, src=0)
                remaining_sgf_files = buf[0]
                FORCE_RELOAD = False
    
    # 今回処理するファイルリストを取り出す（max_files 件まで）
    if len(remaining_sgf_files) < max_files:
        files_to_process = remaining_sgf_files.copy()
    else:
        files_to_process = remaining_sgf_files[:max_files]

    # 毎サイクル、残り/全体 をログ出力
    if TOTAL_SGF_FILES is not None:
        train_logger.info(f"[rank {rank}] pre_train_ds_cycle: remaining {len(remaining_sgf_files)}/{TOTAL_SGF_FILES} files")
    
    # 各 rank に応じてデータを分割（全体の1/nprocsを担当）
    files_to_process = files_to_process[rank::nprocs]
    train_logger.info(f"[rank {rank}] pre_train_ds_cycle: assigned {len(files_to_process)} files.")

    all_samples = []

    train_logger.debug(f"[rank {rank}] pre_train_ds_cycle:(6)")

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
            train_logger.error(f"Error processing file {sgf_file}: {e}")

        # 処理済みファイルを remaining_sgf_files から削除
        remaining_sgf_files.remove(sgf_file)

    train_logger.debug(f"[rank {rank}] pre_train_ds_cycle:(7)")
    
    # 最後に1回だけ進捗を保存（rank==0 のみ）
    if rank == 0:
        save_progress_checkpoint(remaining_sgf_files)

    # サンプルをシャッフルして返却
    random.shuffle(all_samples)
    
    train_logger.debug(f"[rank {rank}] pre_train_ds_cycle:Processed {len(files_to_process)} files; total samples this cycle: {len(all_samples)}")

    #print_tpu_memory()

    return all_samples

def load_training_dataset(sgf_dir, board_size, history_length, resume_flag, augment_all, max_files, rank=0, nprocs=1):
    """
    トレーニング用のデータセットを一度だけ生成し、
    AlphaZeroSGFDatasetPreloaded のインスタンスとして返す関数。
    """
    samples = prepare_train_dataset_cycle(
        sgf_dir, board_size, history_length,
        resume_flag, augment_all, max_files,
        rank=rank, nprocs=nprocs
    )
    dataset = AlphaZeroSGFDatasetPreloaded(samples)

    return dataset

# ==============================
# 訓練ループ用関数（1エポック分）
# ==============================
def train_one_iteration(model, train_loader, optimizer, scheduler, device,
                        epoch, checkpoint_interval, best_policy_accuracy,
                        resume, start_batch_idx, base_seed, rank):
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

    print_interval = 100  # ログ出力間隔（バッチ数）
    accumulated_accuracy = 0.0
    group_batches = 0

    # エポック開始時の時刻を記録
    last_checkpoint_time = time.time()

    # バッチ単位の再開用インデックスを初期化
    last_batch_idx = -1

    # start_batch_idx が全バッチ数を超えていたらリジューム無効化（全スキップ防止）
    total_batches = len(train_loader)
    if resume and start_batch_idx >= total_batches:
        resume = False

    # train_loader 内の各バッチに対してループ
    for i, (boards, target_policies, target_values, target_margins) in enumerate(
            tqdm(train_loader,  desc=f"[rank {rank}]", leave=True, bar_format=bar_fmt)):

        # resume=True のとき、start_batch_idx までのバッチをスキップ
        if resume and i <= start_batch_idx:
            train_logger.debug(f"[rank {rank}] one_it:(1)")
            continue

        # 最後に処理したバッチ番号を記録
        last_batch_idx = i

        train_logger.debug(f"[rank {rank}] one_it:(2)")

        # ── 以降は既存のバッチ処理ロジック ──
        boards = boards.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        target_margins = target_margins.to(device)

        train_logger.debug(f"[rank {rank}] one_it:(3)")

        optimizer.zero_grad()
        pred_policy, (pred_value, pred_margin) = model(boards)
        policy_loss = -torch.sum(target_policies * pred_policy) / boards.size(0)
        value_loss = F.mse_loss(pred_value.view(-1), target_values.view(-1))
        margin_loss = F.mse_loss(pred_margin.view(-1), target_margins.view(-1))
        loss = policy_loss + value_loss_coefficient * value_loss + margin_loss_coefficient * margin_loss

        train_logger.debug(f"[rank {rank}] one_it:(4)")

        loss.backward()
        if USE_TPU:
            # TPUの場合、勾配を同期してパラメータ更新
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer, barrier=True)
        else:
            # GPU/CPU環境では通常の更新
            optimizer.step()

        train_logger.debug(f"[rank {rank}] one_it:(5)")

        # ロスおよび損失項の累積値を加算
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_margin_loss += margin_loss.item()
        num_batches += 1

        train_logger.debug(f"[rank {rank}] one_it:(6)")
        
        # バッチごとに正解率を計算（予測ラベルとターゲットのラベルが一致する割合）
        batch_pred = pred_policy.argmax(dim=1)
        batch_target = target_policies.argmax(dim=1)
        batch_accuracy = (batch_pred == batch_target).float().mean().item()
        overall_correct += (batch_pred == batch_target).sum().item()
        overall_samples += boards.size(0)
        accumulated_accuracy += batch_accuracy
        group_batches += 1

        train_logger.debug(f"[rank {rank}] one_it:(7)")

        # print_interval ごとにログ出力
        if num_batches % print_interval == 0 and rank == 0:
            avg_accuracy = accumulated_accuracy / group_batches
            start_batch = num_batches - group_batches + 1
            end_batch = num_batches
            train_logger.info(f"[rank {rank}] one_it:{start_batch:5d}～{end_batch:5d} policy accuracy : {avg_accuracy:6.4f}")
            accumulated_accuracy = 0.0
            group_batches = 0

        train_logger.debug(f"[rank {rank}] one_it:(8)")

        # 定期チェックポイント保存時に batch_idx を渡す
        if rank == 0:
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                save_checkpoint_nolog(
                    model, optimizer, scheduler,
                    epoch, 0.0, 0.0, best_policy_accuracy,
                    CHECKPOINT_FILE, device,
                    batch_idx=last_batch_idx,
                    base_seed=base_seed
                )
                train_logger.info(f"[rank {rank}] one_it: - checkpoint at epoch {epoch}...")

                # タイマーをリセット
                last_checkpoint_time = current_time

    train_logger.debug(f"[rank {rank}] one_it:(9)")

    if group_batches > 0 and rank == 0:
        avg_accuracy = accumulated_accuracy / group_batches
        train_logger.info(f"[rank {rank}] one_it: Other ({group_batches} batch) policy accuracy : {avg_accuracy:6.4f}")

    if overall_samples > 0 and rank == 0:
        overall_accuracy = overall_correct / overall_samples
        train_logger.info(f"[rank {rank}] one_it: Overall policy accuracy of the latest model state in this training loop: {overall_accuracy:6.4f}")
    else:
        overall_accuracy = 0.0

    avg_loss = total_loss / num_batches
    avg_policy_loss = total_policy_loss / num_batches
    avg_value_loss = value_loss_coefficient * total_value_loss / num_batches
    avg_margin_loss = margin_loss_coefficient * total_margin_loss / num_batches

    # エポックごとの損失や正解率などをログ出力
    if rank == 0:
        train_logger.info(f"[rank {rank}] one_it:Training iteration  total  loss: {avg_loss:.5f}")
        train_logger.info(f"[rank {rank}] one_it:Training iteration  policy loss: {avg_policy_loss:.5f}")
        train_logger.info(f"[rank {rank}] one_it:Training iteration  value  loss: {avg_value_loss:.5f}")
        train_logger.info(f"[rank {rank}] one_it:Training iteration  margin loss: {avg_margin_loss:.5f}")
        train_logger.info(f"[rank {rank}] one_it:Training iteration  policy  acc: {overall_accuracy:.5f}")

    return avg_loss, last_batch_idx

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
from torch.utils.data.distributed import DistributedSampler
def _mp_fn(rank):

    checkpoint_interval = 600 # 600秒
    resume_flag = True
    base_seed = None  # ← 必須：UnboundLocalError防止

    # デバイス初期化
    # デバイス初期化（ordinal = XLA 上の実プロセス番号）
    if USE_TPU:
        import torch_xla.core.xla_model as xm
        import torch.distributed as dist
        device     = xm.xla_device()
        ordinal    = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method='xla://')
        train_logger.info(f"[rank {rank}] _mp_fn: Running on TPU device: {device} | rank = {ordinal}, world_size = {world_size}")
    else:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ordinal    = 0
        world_size = 1
        train_logger.info(f"[rank {rank}] _mp_fn: Running on device: {device} | rank = {ordinal}, world_size = {world_size}")

    # ログ用に ordinal を設定
    # train_logger.rank     = ordinal
    # train_logger.rank       = ordinal
    # train_logger.is_rank0 = (ordinal == 0)
    # train_logger.is_rank0   = (ordinal == 0)

    # アプリケーション開始ログを rank0 のみ出力
    if rank == 0:
        train_logger.info(f"[rank {rank}] === Starting Training and Validation Loop ===")
        # ログ出力で設定内容を確認
        train_logger.info(f"[rank {rank}] ==== Loaded Configuration ====")
        train_logger.info(f"[rank {rank}] Config file: {CONFIG_PATH}")
        train_logger.info(f"[rank {rank}] BOARD_SIZE: {BOARD_SIZE}")
        train_logger.info(f"[rank {rank}] HISTORY_LENGTH: {HISTORY_LENGTH}")
        train_logger.info(f"[rank {rank}] NUM_CHANNELS: {NUM_CHANNELS}")
        train_logger.info(f"[rank {rank}] num_residual_blocks: {num_residual_blocks}")
        train_logger.info(f"[rank {rank}] model_channels: {model_channels}")
        train_logger.info(f"[rank {rank}] batch_size: {batch_size}")
        train_logger.info(f"[rank {rank}] learning_rate: {learning_rate}")
        train_logger.info(f"[rank {rank}] patience: {patience}")
        train_logger.info(f"[rank {rank}] factor: {factor}")
        train_logger.info(f"[rank {rank}] number_max_files: {number_max_files}")
        train_logger.info(f"[rank {rank}] ===============================")

    # テストデータ準備
    test_dataset_pickle = os.path.join(VAL_SGF_DIR, "test_dataset.pkl")
    test_samples = prepare_test_dataset(
        VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, True, test_dataset_pickle
    )
    test_loader = DataLoader(
        AlphaZeroSGFDatasetPreloaded(test_samples),
        batch_size=batch_size, shuffle=False
    )
    train_logger.debug(f"[rank {rank}] _mp_fn:test_loader end.")

    resume_epoch, resume_batch_idx = 0, -1
    mid_epoch_resumed = False
    epoch, best_policy_accuracy = 0, 0.0

    try:
        import torch_xla.core.xla_model as xm

        while True:
            # ── SGFファイルを number_max_files 件ずつ読み込み（重複なくサイクル） ──
            samples = prepare_train_dataset_cycle(
                TRAIN_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH,
                resume_flag, augment_all=True,
                max_files=number_max_files,
                rank=ordinal, nprocs=world_size
            )
            # 初回のみ resume_flag をリセット
            resume_flag = False
            # Dataset 化（rank ごとに分割済み）
            training_dataset = AlphaZeroSGFDatasetPreloaded(samples)
            train_logger.info(f"[rank {rank}] _mp_fn: prepared {len(samples)} samples this epoch")

            # モデルとオプティマイザ・スケジューラを構築
            model = EnhancedResNetPolicyValueNetwork(
                BOARD_SIZE, model_channels, num_residual_blocks, NUM_CHANNELS
            ).to(device)
            train_logger.debug(f"[rank {rank}] _mp_fn:model instance to {device}")

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=patience, factor=factor
            )

            # チェックポイント読み込み
            resume_epoch, best_policy_accuracy, resume_batch_idx, loaded_seed = load_checkpoint(
                model, optimizer, scheduler, CHECKPOINT_FILE, device
            )

            # optimizer から復元された現在の学習率を取得
            if ordinal == 0:
                # ■■■ lrを変更 ■■■
                # optimizer.param_groups[0]['lr'] = 0.0001

                current_lr = optimizer.param_groups[0]["lr"]
                train_logger.info(f"[rank {rank}] =========== params ============")
                train_logger.info(f"[rank {rank}] learning_rate (from optimizer): {current_lr:.6f}")
                train_logger.info(f"[rank {rank}] patience: {patience}")
                train_logger.info(f"[rank {rank}] factor: {factor}")
                train_logger.info(f"[rank {rank}] number_max_files: {number_max_files}")
                train_logger.info(f"[rank {rank}] resume_epoch: {resume_epoch}, resume_batch: {resume_batch_idx}")
                train_logger.info(f"[rank {rank}] best_policy_accuracy (from checkpoint): {best_policy_accuracy:.5f}")
                if loaded_seed is not None:
                    train_logger.info(f"[rank {rank}] loaded_seed: {loaded_seed}")
                train_logger.info(f"[rank {rank}] ===============================")

            base_seed = loaded_seed if loaded_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
            epoch = resume_epoch

            seed = base_seed + epoch * world_size + ordinal
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if ordinal == 0:
                train_logger.info(
                    f"[rank {rank}] _mp_fn:Initial best_policy_accuracy: {best_policy_accuracy:.5f}, resume at epoch={resume_epoch}, batch={resume_batch_idx}"
                )

            # データセットを同期的にロード
            # training_dataset は既に broadcast→Dataset化済みなので再ロード不要
            train_logger.debug(f"[rank {rank}] _mp_fn: using broadcasted dataset")

            # DataLoader：シャッフルしてバッチ生成
            num_workers = 1
            train_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                prefetch_factor=None if num_workers == 0 else 1,
                persistent_workers=True if num_workers > 0 else False
            )
            train_logger.info(f"[rank {rank}] _mp_fn:Start epoch {epoch}, {len(train_loader)} batches")

            # バッチ単位での中断復元判定
            do_resume = (not mid_epoch_resumed) and (epoch == resume_epoch) and (resume_batch_idx >= 0)
            start_idx = resume_batch_idx if do_resume else -1

            train_logger.debug(f"[rank {rank}] _mp_fn:train_one_iteration start.")
            avg_loss, last_batch_idx = train_one_iteration(
                model, train_loader, optimizer, scheduler,
                device, epoch, checkpoint_interval, best_policy_accuracy,
                do_resume, start_idx, base_seed, ordinal)

            mid_epoch_resumed = mid_epoch_resumed or do_resume
            train_logger.info(f"[rank {rank}] _mp_fn:train_one_iteration end. epoch:{epoch}")
            
            # train_loader のみ破棄。training_dataset はループ外で再利用する
            del train_loader
            gc.collect()

            # 評価およびモデル保存
            policy_accuracy = None
            if ordinal == 0:
                policy_accuracy = validate_model(model, test_loader, device)
                if policy_accuracy > best_policy_accuracy:
                    best_policy_accuracy = save_best_model(
                        model, policy_accuracy, device, best_policy_accuracy
                    )
                else:
                    save_inference_model(model, device, f"{INFERENCE_MODEL_PREFIX}_tmp.pt")

            if ordinal == 0:
                train_logger.info(
                    f"[rank {rank}] _mp_fn:Epoch {epoch} - Before scheduler.step(): lr = {optimizer.param_groups[0]['lr']:.8f}"
                )
            
            if ordinal == 0:
                scheduler.step(policy_accuracy)
            
            if ordinal == 0:
                train_logger.info(
                    f"[rank {rank}] _mp_fn:Epoch {epoch} - After  scheduler.step(): lr = {optimizer.param_groups[0]['lr']:.8f}"
                )

            if ordinal == 0:
                # チェックポイント保存
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    0.0, 0, best_policy_accuracy,
                    CHECKPOINT_FILE, device,
                    batch_idx=last_batch_idx,
                    base_seed=base_seed
                )
            if ordinal == 0:
                train_logger.info(f"[rank {rank}] _mp_fn:Iteration completed. Restarting next iteration...\n")
            
            epoch += 1
    finally:
        pass

# ==============================
# main処理
# ==============================
def main():
    # ── CLI引数から渡された値と適用後のグローバル変数を出力 ──
    train_logger.info(f"PREFIX: {PREFIX}")
    train_logger.info(f"FORCE_RELOAD: {FORCE_RELOAD}")
    train_logger.info(f"TVAL_SGF_DIR {VAL_SGF_DIR}")
    train_logger.info(f"TRAIN_SGF_DIR {TRAIN_SGF_DIR}")
    train_logger.info(f"MODEL_OUTPUT_DIR {MODEL_OUTPUT_DIR}")
    train_logger.info(f"CHECKPOINT_FILE: {CHECKPOINT_FILE}")
    train_logger.info(f"PROGRESS_CHECKPOINT_FILE: {PROGRESS_CHECKPOINT_FILE}")

    # ここに各種初期化、設定ロード、ロガー設定など
    if USE_TPU:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=())  # XLA runtime未初期化状態で呼び出す必要あり
    else:
        # TPUでない場合、シングルプロセスで _mp_fn を呼び出す
        _mp_fn(0)

if __name__ == "__main__":
    main()


