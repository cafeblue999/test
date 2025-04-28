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
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from dataset import (
    AlphaZeroSGFDatasetPreloaded,
    prepare_test_dataset,
    save_checkpoint,
    save_checkpoint_nolog,
    validate_model,
    save_best_model,
    save_inference_model,
    process_sgf_to_samples_from_text,
    load_checkpoint
)

from config import PREFIX, USE_TPU, FORCE_RELOAD, TRAIN_SGF_DIR, VAL_SGF_DIR, MODEL_OUTPUT_DIR, INFERENCE_MODEL_PREFIX, CHECKPOINT_FILE, bar_fmt, get_logger, tqdm_kwargs

from utils import BOARD_SIZE, HISTORY_LENGTH, NUM_CHANNELS, num_residual_blocks, model_channels, batch_size, learning_rate, patience, factor, number_max_files, number_proc_files, CONFIG_PATH, val_interval

train_logger = get_logger()

# ==============================
# 訓練ループ用関数（1エポック分）
# ==============================
def train_one_iteration(
    model, train_loader, optimizer, scheduler, device,
    local_loop_cnt, checkpoint_interval, best_policy_accuracy, rank
):
    model.train()
    total_loss = total_policy = total_value = total_margin = 0.0
    overall_correct = overall_samples = 0
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 10)  # 10% ごとにログ

    for i, (boards, target_policies, target_values, target_margins) in enumerate(train_loader, start=1):
        
        boards = boards.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)
        target_margins = target_margins.to(device)

        # 順伝播→損失計算
        pred_policy, (pred_value, pred_margin) = model(boards)
        policy_loss = -torch.sum(target_policies * pred_policy) / boards.size(0)
        value_loss  = F.mse_loss(pred_value.view(-1), target_values.view(-1))
        margin_loss = F.mse_loss(pred_margin.view(-1), target_margins.view(-1))
        loss = policy_loss + 0.05 * value_loss + 0.0001 * margin_loss

        if not torch.isfinite(loss):
            train_logger.error(f"[rank {rank}] Invalid loss: {loss}. Skipping this batch.")
            continue

        # 逆伝播→更新
        optimizer.zero_grad()
        loss.backward()
        
        if USE_TPU:
            xm.mark_step()  # lazy実行を明示的に発火させる
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()

        # メトリクス集計
        total_loss += loss.item()
        total_policy += policy_loss.item()
        total_value  += value_loss.item()
        total_margin += margin_loss.item()
        batch_pred = pred_policy.argmax(dim=1)
        batch_true = target_policies.argmax(dim=1)
        overall_correct += (batch_pred == batch_true).sum().item()
        overall_samples += boards.size(0)

        # 進捗ログ
        if i % log_interval == 0 or i == total_batches:
            train_logger.info(
                f"[rank {rank}] Epoch {local_loop_cnt}: processed {i:3d}/{total_batches:3d} batches {i*100/total_batches:3.0f}%)")

    # エポック終了ログ
    if rank == 0:
        avg_loss = total_loss / len(train_loader)
        acc = overall_correct / overall_samples
        train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt} done. loss={avg_loss:.5f}, acc={acc:.5f}")
    
    train_logger.debug(f"[rank {rank}] train_one_iteration done. Returning control to _mp_fn")
    
    return total_loss / len(train_loader)

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
from torch.utils.data.distributed import DistributedSampler
def _mp_fn(rank):
   
    # チェックポイント保存間隔（秒）
    checkpoint_interval = 600

    # デバイス初期化
    if USE_TPU:
        device     = xm.xla_device()
        ordinal    = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method='xla://')
    else:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ordinal    = 0
        world_size = 1

    train_logger.info(f"[rank {rank}] Running on device: {device} | ordinal = {ordinal}, world_size = {world_size}")

    if ordinal == 0:
        # テストデータ準備（ループ外で一度だけ）
        test_dataset_pickle = os.path.join(VAL_SGF_DIR, "test_dataset.pkl")
        test_samples = prepare_test_dataset(
            VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, augment_all=True, output_file=test_dataset_pickle
        )
        test_loader = DataLoader(
            AlphaZeroSGFDatasetPreloaded(test_samples),
            batch_size=batch_size, shuffle=False
        )
        train_logger.info(f"[rank {rank}] Test loader ready. {len(test_loader.dataset)} samples")

    # 最良精度の初期化
    best_policy_accuracy = 0.0

    # 全ファイル読み込み
    all_files = [
        os.path.join(TRAIN_SGF_DIR, f)
        for f in sorted(os.listdir(TRAIN_SGF_DIR))
        if f.endswith('.sgf') and "analyzed" not in f.lower()
    ]
    train_logger.info(f"[rank {rank}] Available {len(all_files)} SGF files (shared across all ranks)")
    
    local_loop_cnt = 0

    while True:
        # epoch と rank を組み合わせたシード
        base_seed = int(time.time_ns() % (2**32))
        seed = base_seed + local_loop_cnt * world_size + ordinal
        random.seed(seed)
        torch.manual_seed(seed)

        # モデル／オプティマイザ／スケジューラ生成
        model = EnhancedResNetPolicyValueNetwork(
            BOARD_SIZE, model_channels, num_residual_blocks, NUM_CHANNELS
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor
        )

        # 既存チェックポイントの読み込み
        epoch, best_policy_accuracy, last_batch_idx, base_seed = load_checkpoint(
            model, optimizer, scheduler,
            CHECKPOINT_FILE, device
        )
        # モデルを改めてXLAデバイスに配置（復元後）
        model.to(device)   
          
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            train_logger.info(f"[rank {rank}] =========== params ============")
            train_logger.info(f"[rank {rank}] epoch: {local_loop_cnt}")
            train_logger.info(f"[rank {rank}] learning_rate (from optimizer): {current_lr:.8f}")
            train_logger.info(f"[rank {rank}] patience: {patience}")
            train_logger.info(f"[rank {rank}] factor: {factor}")
            train_logger.info(f"[rank {rank}] number_max_files: {number_max_files}")
            train_logger.info(f"[rank {rank}] best_policy_accuracy (from checkpoint): {best_policy_accuracy:.5f}")
            train_logger.info(f"[rank {rank}] val_interval: {val_interval}")
            train_logger.info(f"[rank {rank}] ===============================")
        
        # 各 epoch ごとにランダムに sampling（全rank共通リストから）
        if len(all_files) >= number_max_files:
            selected = random.sample(all_files, k=number_max_files)
        else:
            selected = random.choices(all_files, k=number_max_files)
        train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt}: sampled {len(selected)} files (random from shared)")

        # SGF→サンプル生成
        samples = []
        for sgf_file in selected:
            try:
                with open(sgf_file, "r", encoding="utf-8") as f:
                    sgf_src = f.read()
                samples.extend(process_sgf_to_samples_from_text(
                    sgf_src, BOARD_SIZE, HISTORY_LENGTH, augment_all=True
                ))
            except Exception as e:
                train_logger.error(f"[rank {rank}] Error processing {sgf_file}: {e}")
        if len(samples) == 0:
            train_logger.warning(f"[rank {rank}] No samples generated. Skipping epoch {local_loop_cnt}.")
            local_loop_cnt += 1
            continue

        #  DistributedSampler を使った DataLoader 組み立て
        train_dataset = AlphaZeroSGFDatasetPreloaded(samples)

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=ordinal,
            shuffle=True,
            seed=seed,
            drop_last=True
        )

        # エポックごとにシードを変える
        train_sampler.set_epoch(local_loop_cnt)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=False
        )
        train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt}: {len(train_loader)} batches")

        # TPU 向けにデバイスローダーを作成
        train_device_loader = MpDeviceLoader(train_loader, device)

        # １エポック分の訓練（デバイスローダーを渡す）
        avg_loss = train_one_iteration(
            model, train_device_loader, optimizer, scheduler,
             device, local_loop_cnt, checkpoint_interval,
             best_policy_accuracy, rank
        )
        train_logger.debug(f"[rank {rank}] train_one_iteration returned, evaluating if validation is needed")

        local_loop_cnt += 1

        # 検証とモデル保存（ordinal=0 のみ）
        if ordinal == 0 and local_loop_cnt % val_interval == 0:
            train_logger.info(f"[rank {rank}] Starting validation after epoch {local_loop_cnt}")
            model.eval()  # ドロップアウトや BN を評価モードに
            # グラフを保持せずに検証を実行
            with torch.no_grad():
                policy_accuracy = validate_model(model, test_loader, device)
            # 一時変数解放
            model.train()

            if policy_accuracy > best_policy_accuracy:
                best_policy_accuracy = save_best_model(
                    model, policy_accuracy, device, best_policy_accuracy
                )
            else:
                save_inference_model(model, device, f"{INFERENCE_MODEL_PREFIX}_tmp.pt")

            # scheduler.step() 実行前の学習率を表示
            before_lr = optimizer.param_groups[0]["lr"]
            train_logger.info(f"[rank {rank}] _mp_fn: Before scheduler.step(): learning rate = {before_lr:.8f}")           
            # 学習率調整
            scheduler.step(policy_accuracy)
            # scheduler.step() 実行後の学習率を表示
            after_lr = optimizer.param_groups[0]["lr"]
            train_logger.info(f"[rank {rank}] _mp_fn: After  scheduler.step(): learning rate = {after_lr:.8f}")

            # チェックポイント保存
            save_checkpoint(
                model, optimizer, scheduler,
                local_loop_cnt,        # epoch 保存
                0.0,                   # val_loss は使用せず
                0,                     # epochs_no_improve は使用せず
                best_policy_accuracy,
                CHECKPOINT_FILE, device,
                batch_idx=-1,          # バッチ再開なし
                base_seed=None
            )
            train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt} completed. best_acc={best_policy_accuracy:.5f}")

        del model, optimizer, scheduler, samples, train_dataset, train_loader, train_device_loader
        gc.collect()
        xm.mark_step()

# ==============================
# main処理
# ==============================
def main():
    # ── CLI引数から渡された値と適用後のグローバル変数を出力 ──
    train_logger.info(f"PREFIX: {PREFIX}")
    train_logger.info(f"FORCE_RELOAD: {FORCE_RELOAD}")
    train_logger.info(f"VAL_SGF_DIR {VAL_SGF_DIR}")
    train_logger.info(f"TRAIN_SGF_DIR {TRAIN_SGF_DIR}")
    train_logger.info(f"MODEL_OUTPUT_DIR {MODEL_OUTPUT_DIR}")
    train_logger.info(f"CHECKPOINT_FILE: {CHECKPOINT_FILE}")

    # ここに各種初期化、設定ロード、ロガー設定など
    if USE_TPU:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=())  # XLA runtime未初期化状態で呼び出す必要あり
    else:
        # TPUでない場合、シングルプロセスで _mp_fn を呼び出す
        _mp_fn(0)

if __name__ == "__main__":
    main()


