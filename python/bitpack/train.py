import os
import zipfile
import random
import time
import datetime
import gc
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch_xla.core.xla_model as xm
import torch.distributed as dist
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from model import EnhancedResNetPolicyValueNetwork

from dataset import (
    AlphaZeroSGFDatasetPreloaded,
    prepare_test_dataset,
    save_checkpoint,
    save_checkpoint_nolog,
    validate_model,
    save_best_acc_model,
    save_best_loss_model,
    save_inference_model,
    process_sgf_to_samples_from_text,
    load_checkpoint
)

from config import PREFIX, USE_TPU, FORCE_RELOAD, BASE_DIR, TRAIN_SGF_DIR, TRAIN_SGFS_ZIP, VAL_SGF_DIR, MODEL_OUTPUT_DIR, INFERENCE_MODEL_PREFIX, CHECKPOINT_FILE, bar_fmt, get_logger, tqdm_kwargs, LOG_DIR, LOSS_LOG_DIR, JST, COUNTS_FILE, TEST_DATASET_PKL

from utils import BOARD_SIZE, HISTORY_LENGTH, NUM_CHANNELS, num_residual_blocks, model_channels, learning_rate, patience, factor, number_max_files, number_proc_files, CONFIG_PATH, val_interval, w_policy_loss, w_value_loss, argument, train_batch_size, test_batch_size

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

train_logger = get_logger()

# ==============================
# 重み付けサンプリング
# ==============================
def weighted_file_sample(all_files, counts_file, number_max_files):
    """
    all_files: List of (zip_path, entry_name) tuples
    counts_file: path to pickle with {file_id: count}
    number_max_files: number of files to sample
    """
    # 1) 保存済みカウントをロード
    if os.path.exists(counts_file):
        with open(counts_file, "rb") as f:
            counts = pickle.load(f)
    else:
        counts = {}

    # 2) ファイルIDリストを作成
    file_ids = [f"{zp}:{ename}" for zp, ename in all_files]

    # 3) 重みを計算（べき乗で浅いカウントを強調）
    exponent = 3.0
    weights = [
        1.0 / ((counts.get(fid, 0) + 1) ** exponent)
        for fid in file_ids
    ]
    total_w = float(sum(weights))
    probs   = [w / total_w for w in weights]

    # 4) サンプリング数の決定
    k = min(len(all_files), number_max_files)

    # 5) 未使用ファイル（count==0）の優先選択
    unused_idxs = [i for i, fid in enumerate(file_ids) if counts.get(fid, 0) == 0]
    if unused_idxs:
        # 5-1) 未使用が k 個以上あれば、その中から k 個を均等サンプリング
        if len(unused_idxs) >= k:
            chosen = np.random.choice(unused_idxs, size=k, replace=False)
            return [all_files[i] for i in chosen]

        # 5-2) 未使用が不足する場合：まず全て取得し、残りを重み付きで補充
        selected = list(unused_idxs)
        remaining_k = k - len(selected)

        rem_idxs    = [i for i in range(len(all_files)) if i not in selected]
        rem_weights = [weights[i] for i in rem_idxs]
        rem_total   = float(sum(rem_weights))
        rem_probs   = [w / rem_total for w in rem_weights]

        fill = np.random.choice(rem_idxs, size=remaining_k, replace=False, p=rem_probs)
        selected.extend(fill.tolist())
        return [all_files[i] for i in selected]

    # 6) 未使用ファイルがなければ、通常の重み付きサンプリング
    idxs = np.random.choice(len(all_files), size=k, replace=False, p=probs)

    return [all_files[i] for i in idxs]

# ==============================
# 訓練ループ用関数（1エポック分）
# ==============================
def train_one_iteration(model, train_loader, optimizer, device, local_loop_cnt, rank):

    model.train()

    total_loss = total_policy = total_value = 0.0
    
    # ログ用ファイルをバッチ外で一度だけオープン
    policy_log_f = None
    value_log_f  = None

    if rank == 0:
        policy_log_f = open(
            os.path.join(LOSS_LOG_DIR, "weighted_policy_loss.log"),
            "a", encoding="utf-8"
        )
        value_log_f = open(
            os.path.join(LOSS_LOG_DIR, "weighted_value_loss.log"),
            "a", encoding="utf-8"
        )
    
    overall_correct = overall_samples = 0
    total_batches = len(train_loader)
    log_interval = max(1, total_batches // 10)  # 10% ごとにログ

    for i, batch in enumerate(train_loader, start=1):
        # まずは必ずクリアしてからバッチ処理
        optimizer.zero_grad()
        # unpack batch
        boards, target_policies, target_values = batch

        # 非同期転送
        # TPU: bfloat16／long にキャスト
        boards          = boards.to(device, non_blocking=True, dtype=torch.bfloat16)
        target_policies = target_policies.to(device, non_blocking=True, dtype=torch.long)
        target_values   = target_values.to(device, non_blocking=True, dtype=torch.bfloat16)

        # 順伝播→損失計算（損失定義を検証時と同じクロスエントロピーに統一）
        # TPU: bfloat16 演算でフォワード→ロス計算
        pred_policy, pred_value = model(boards)
        target_labels = target_policies.argmax(dim=1)
        # loss は float32 にキャストして計算
        policy_loss = F.cross_entropy(pred_policy.float(), target_labels)
        value_loss  = F.mse_loss(pred_value.view(-1).float(), target_values.view(-1).float())
        loss = w_policy_loss * policy_loss + w_value_loss * value_loss

        # weighted loss のログ追記 ──
        if rank == 0:
            ts = datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
            policy_log_f.write(
                f"{ts},{local_loop_cnt},{i},{(w_policy_loss*policy_loss).item():.6f}\n"
            )
            value_log_f.write(
                f"{ts},{local_loop_cnt},{i},{(w_value_loss*value_loss).item():.6f}\n"
            )

        if not torch.isfinite(loss):
            train_logger.error(f"[rank {rank}] Invalid loss: {loss}. Skipping this batch.")
            continue

        # TPU: 逆伝播→明示的ステップ→更新
        loss.backward()
        xm.mark_step()
        xm.optimizer_step(optimizer, barrier=True)

        # メトリクス集計
        total_loss += loss.item()
        total_policy += policy_loss.item()
        total_value  += value_loss.item()
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
        train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt}: iteration done. loss={avg_loss:.5f}, acc={acc:.5f}")
        policy_log_f.close()
        value_log_f.close()
    
    return total_loss / len(train_loader)

# ==============================
# TPU分散環境で動作するメイン処理
# ==============================
def _mp_fn(rank):
   
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
        test_dataset_pickle = TEST_DATASET_PKL
        test_samples = prepare_test_dataset(
            VAL_SGF_DIR, BOARD_SIZE, HISTORY_LENGTH, augment_all=0, output_file=test_dataset_pickle
        )
        test_loader = DataLoader(
            AlphaZeroSGFDatasetPreloaded(test_samples, []),
            batch_size=test_batch_size, shuffle=False
        )
        train_logger.info(f"[rank {rank}] Test loader ready. {len(test_loader.dataset)} samples")

    # 最良精度の初期化
    best_policy_accuracy = 0.0

    # 全ファイル読み込み (ZIP をループ外でオープン)
    zip_path = TRAIN_SGFS_ZIP
    zip_ref  = zipfile.ZipFile(zip_path, 'r')
    all_files = []
    # .sgfファイルのうち "analyzed" を含まないものを一覧化
    sgf_files = [
        (zip_path, name)
        for name in zip_ref.namelist()
        if name.endswith('.sgf') and "analyzed" not in name.lower()
    ]
    all_files.extend(sgf_files)           
    train_logger.info(f"[rank {rank}] Available {len(all_files)} SGF files (shared across all ranks)")
    
    # all_files をシャッフル(すでにtrain.zip作成時にファイル単位でシャッフルされているが念の為)
    random.shuffle(all_files)
    
    # 全ファイルID一覧を作成（各要素を "zip_path:entry_name" 形式の文字列に）
    all_file_ids = [f"{zp}:{entry}" for zp, entry in all_files]

    local_loop_cnt = 0

    while True:
        # epoch と rank を組み合わせたシード
        base_seed = int(time.time_ns() % (2**32))
        seed = base_seed + local_loop_cnt * world_size + ordinal
        random.seed(seed)
        torch.manual_seed(seed)

        # モデル／オプティマイザ／スケジューラ生成
        model = EnhancedResNetPolicyValueNetwork(
             BOARD_SIZE,
             NUM_CHANNELS,         # in_channels: 入力テンソルのチャネル数
             model_channels,       # num_channels: ネットワーク内部のチャネル数
             num_residual_blocks   # num_blocks: ResidualBlock の数
        ).to(device, dtype=torch.bfloat16)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor
        )

        # 既存チェックポイントの読み込み
        best_total_loss, best_policy_accuracy = load_checkpoint(model, optimizer, scheduler, CHECKPOINT_FILE, device)
        # モデルを改めてXLAデバイスに配置（復元後）
        model.to(device)   
          
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            mode = scheduler.state_dict()['mode']
            chk_factor = scheduler.state_dict()['factor']
            chk_patience = scheduler.state_dict()['patience']
            bad_epochs = scheduler.num_bad_epochs
            train_logger.info(f"[rank {rank}] =============== runtime params ============")
            train_logger.info(f"[rank {rank}] epoch                     : {local_loop_cnt}")
            train_logger.info(f"[rank {rank}] train_batch_size          : {train_batch_size}")
            train_logger.info(f"[rank {rank}] test_batch_size           : {test_batch_size}")
            train_logger.info(f"[rank {rank}] train argument            : {argument}")
            train_logger.info(f"[rank {rank}] optimizer lr_rate    (chk): {current_lr:.8f}")
            train_logger.info(f"[rank {rank}] scheduler patience   (chk): {chk_patience}")
            train_logger.info(f"[rank {rank}] scheduler factor     (chk): {chk_factor}")
            train_logger.info(f"[rank {rank}] scheduler mode       (chk): {mode}")
            train_logger.info(f"[rank {rank}] scheduler bad_epochs (chk): {bad_epochs}")
            train_logger.info(f"[rank {rank}] best_total_loss      (chk): {best_total_loss:.5f}")
            train_logger.info(f"[rank {rank}] best_policy_accuracy (chk): {best_policy_accuracy:.5f}")
            train_logger.info(f"[rank {rank}] number_max_files          : {number_max_files}")
            train_logger.info(f"[rank {rank}] val_interval              : {val_interval}")
            train_logger.info(f"[rank {rank}] w_policy_loss             : {w_policy_loss}")
            train_logger.info(f"[rank {rank}] w_value_loss              : {w_value_loss}")
            train_logger.info(f"[rank {rank}] ===========================================")
        
        # ① 全rank共通でnumber_max_filesファイルを全SGFファイルより選択
        all_selected = weighted_file_sample(all_files, COUNTS_FILE, number_max_files)

        # ② 各rankはその中のスライスだけ処理（チャンク分割）
        selected = all_selected[ordinal::world_size]

        train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt}: sampled {len(selected)} files")

        # SGF→サンプル生成（元ファイル情報も一緒に保持）
        samples = []
        # selected は [(zip_path, entry_name), ...]
        for zip_path, entry_name in selected:
            sgf_src = zip_ref.read(entry_name).decode('utf-8')
            file_id = f"{zip_path}:{entry_name}"
            for inp, pol, val in process_sgf_to_samples_from_text(
                    sgf_src, BOARD_SIZE, HISTORY_LENGTH, augment_all=argument):
                # タプルの末尾に file_id を追加
                samples.append((inp, pol, val, file_id))

        if not samples:
            train_logger.warning(f"[rank {rank}] No samples generated. Skipping epoch {local_loop_cnt}.")
            local_loop_cnt += 1
            continue
        else:
            train_logger.info(f"[rank {rank}] samples generated.")
        
        # ③ サンプルリストを Dataset に渡す際、file_list に全ファイルIDを指定
        train_dataset = AlphaZeroSGFDatasetPreloaded(samples, all_file_ids)

        # ここで train_dataset が定義済みなのでカウントできる
        if ordinal == 0:
            for zip_path, entry_name in all_selected:
                file_id = f"{zip_path}:{entry_name}"
                train_dataset.file_process_counts[file_id] += 1

        # ④ DistributedSampler は使わずに、DataLoader の shuffle だけでランダム化
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,               # ← ここだけで十分
            drop_last=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=False
        )

        # TPU 向けにデバイスローダーを作成
        train_device_loader = MpDeviceLoader(train_loader, device)

        # １エポック分の訓練（デバイスローダーを渡す）
        avg_loss = train_one_iteration(model, train_device_loader, optimizer, device, local_loop_cnt, rank)
        train_logger.info(f"[rank {rank}] train_one_iteration end, avg_loss: {avg_loss:0.5f}")

        local_loop_cnt += 1

        #xm.rendezvous("sync")

        # 検証とモデル保存（ordinal=0 のみ）
        if ordinal == 0 and local_loop_cnt % val_interval == 0:
            train_logger.info(f"[rank {rank}] Starting validation after epoch {local_loop_cnt - 1}")
            
            # 検証を実行
            with torch.no_grad():
                policy_acc, avg_value_loss, total_loss = validate_model(model, test_loader, device)  # margin項なしに合わせて修正

            if total_loss < best_total_loss:
                train_logger.info(f"[rank {rank}] _mp_fn: total_loss {best_total_loss:.5f} → {total_loss:.5f}" )
                best_total_loss = total_loss
                save_best_loss_model(model, total_loss, device)
            #else:
            #    save_inference_model(model, device, f"{INFERENCE_MODEL_PREFIX}_loss_tmp.pt")

            if policy_acc > best_policy_accuracy:
                train_logger.info(f"[rank {rank}] _mp_fn: total_acc {best_policy_accuracy:.5f} → {policy_acc:.5f}" )
                best_policy_accuracy = policy_acc
                save_best_acc_model(model, policy_acc, device)
            #else:
            #    save_inference_model(model, device, f"{INFERENCE_MODEL_PREFIX}_acc_tmp.pt")

            # scheduler.step() 実行前の学習率を表示
            before_lr = optimizer.param_groups[0]["lr"]
            train_logger.info(f"[rank {rank}] _mp_fn: Before scheduler.step(): learning rate = {before_lr:.8f}") 

            # 学習率調整
            scheduler.step(policy_acc)
            #scheduler.step(total_loss)

            # scheduler.step() 実行後の学習率を表示
            after_lr = optimizer.param_groups[0]["lr"]
            train_logger.info(f"[rank {rank}] _mp_fn: After  scheduler.step(): learning rate = {after_lr:.8f}")

            # チェックポイント保存
            save_checkpoint(model, optimizer, scheduler, best_total_loss, best_policy_accuracy, CHECKPOINT_FILE)
            train_logger.info(f"[rank {rank}] Epoch {local_loop_cnt - 1} completed.")

            # エポック終了後に未処理ファイル数をログ＆カウント保存
            try:
                zero_count = sum(1 for cnt in train_dataset.file_process_counts.values() if cnt == 0)
                total_files = len(train_dataset.file_list)
                train_logger.info(f"[rank {rank}] remaining files: {zero_count} / {total_files}")
                train_dataset.save_file_counts()
            except Exception as e:
                train_logger.warning(f"[rank {rank}] fail to save count file: {e}")
        
        del model, optimizer, scheduler, samples, train_dataset, train_loader, train_device_loader
        gc.collect()
        xm.mark_step()
        
    # ループ外で ZIP をクローズ(上記は現状では無限ループなので到達しない)
    zip_ref.close()

# ==============================
# main処理
# ==============================
def main():
    # ── CLI引数から渡された値と適用後のグローバル変数を出力 ──
    train_logger.info(f"PREFIX           : {PREFIX}")
    train_logger.info(f"FORCE_RELOAD     : {FORCE_RELOAD}")
    train_logger.info(f"BASE_DIR         : {BASE_DIR}")
    train_logger.info(f"VAL_SGF_DIR      : {VAL_SGF_DIR}")
    train_logger.info(f"TRAIN_SGF_DIR    : {TRAIN_SGF_DIR}")
    train_logger.info(f"MODEL_OUTPUT_DIR : {MODEL_OUTPUT_DIR}")
    train_logger.info(f"CHECKPOINT_FILE  : {CHECKPOINT_FILE}")
    train_logger.info(f"=========== ini ============")
    train_logger.info(f"train_batch_size : {train_batch_size}")
    train_logger.info(f"test_batch_size  : {test_batch_size}")
    train_logger.info(f"argument         : {argument}")
    train_logger.info(f"number_max_files : {number_max_files }")
    train_logger.info(f"patience         : {patience}")
    train_logger.info(f"factor           : {factor}")
    train_logger.info(f"val_interval     : {val_interval}")
    train_logger.info(f"w_policy_loss    : {w_policy_loss}")
    train_logger.info(f"w_value_loss     : {w_value_loss}")
    train_logger.info(f"============================")

    # ここに各種初期化、設定ロード、ロガー設定など
    if USE_TPU:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=())  # XLA runtime未初期化状態で呼び出す必要あり
    else:
        # TPUでない場合、シングルプロセスで _mp_fn を呼び出す
        _mp_fn(0)

if __name__ == "__main__":
    main()


