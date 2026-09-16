from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

SCRIPT_FOLDER = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_FOLDER / "CHB_MIT")
)

from model import Attentive_RNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log(message, log_file):
    print(message, flush=True)

    with open(
        log_file,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(str(message) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1111
    )

    args = parser.parse_args()

    cache_folder = Path(args.cache_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    log_file = output_folder / "training_console_log.txt"

    if log_file.exists():
        log_file.unlink()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    signal_file = (
        cache_folder / "raw_training_signals.npy"
    )

    label_file = (
        cache_folder / "raw_training_labels.npy"
    )

    if not signal_file.is_file():
        raise FileNotFoundError(signal_file)

    if not label_file.is_file():
        raise FileNotFoundError(label_file)

    write_log(
        "=" * 70,
        log_file
    )

    write_log(
        "ORIGINAL ARNN WITH RAW CHB-MIT TRAINING DATA",
        log_file
    )

    write_log(
        "=" * 70,
        log_file
    )

    write_log(
        f"Device: {device}",
        log_file
    )

    if torch.cuda.is_available():
        write_log(
            f"GPU: {torch.cuda.get_device_name(0)}",
            log_file
        )

    write_log(
        f"Random seed: {args.seed}",
        log_file
    )

    write_log(
        f"Epochs: {args.epochs}",
        log_file
    )

    write_log(
        f"Batch size: {args.batch_size}",
        log_file
    )

    write_log(
        f"Initial learning rate: {args.learning_rate}",
        log_file
    )

    write_log(
        "Loading the training cache...",
        log_file
    )

    signals = np.load(signal_file).astype(
        np.float32,
        copy=False
    )

    labels = np.load(label_file).astype(
        np.float32,
        copy=False
    )

    if signals.shape != (1886, 1024, 18):
        raise ValueError(
            f"Unexpected signal shape: {signals.shape}"
        )

    if labels.shape != (1886, 1):
        raise ValueError(
            f"Unexpected label shape: {labels.shape}"
        )

    signal_tensor = torch.from_numpy(signals)
    label_tensor = torch.from_numpy(labels)

    training_dataset = TensorDataset(
        signal_tensor,
        label_tensor
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    write_log(
        f"Training signal shape: {signals.shape}",
        log_file
    )

    write_log(
        f"Training label shape: {labels.shape}",
        log_file
    )

    write_log(
        "Nonseizure training segments: "
        f"{int((labels == 0).sum())}",
        log_file
    )

    write_log(
        "Seizure training segments: "
        f"{int((labels == 1).sum())}",
        log_file
    )

    # Original ARNN architecture settings.
    model = Attentive_RNN(
        d_dim=18,
        embed_dim=40,
        seq_len=1024,
        dim_head=10,
        heads=4,
        num_state_vectors=64,
        time_steps=16,
        num_class=1,
        qk_rmsnorm=True,
        rotary_pos_emb=True
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    write_log(
        f"Model parameters: {parameter_count:,}",
        log_file
    )

    criterion = nn.BCELoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    history = []
    total_training_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_start = time.time()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for batch_number, (data, target) in enumerate(
            training_loader,
            start=1
        ):
            data = data.to(
                device,
                non_blocking=True
            )

            target = target.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            probability = model(data)
            loss = criterion(probability, target)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * data.size(0)
            )

            predictions = (
                probability >= 0.5
            ).float()

            all_labels.extend(
                target.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(int)
                .tolist()
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(int)
                .tolist()
            )

            if (
                batch_number == 1
                or batch_number % 10 == 0
                or batch_number == len(training_loader)
            ):
                write_log(
                    f"Train Epoch: {epoch} "
                    f"[Batch {batch_number}/"
                    f"{len(training_loader)}] "
                    f"Loss: {loss.item():.6f}",
                    log_file
                )

        epoch_seconds = time.time() - epoch_start
        total_training_seconds += epoch_seconds

        epoch_loss = (
            total_loss / len(training_dataset)
        )

        epoch_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        epoch_f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "Epoch": epoch,
                "Training_Loss": epoch_loss,
                "Training_Accuracy": epoch_accuracy,
                "Training_F1": epoch_f1,
                "Learning_Rate": current_learning_rate,
                "Epoch_Seconds": epoch_seconds,
            }
        )

        write_log(
            (
                f"Epoch {epoch} completed | "
                f"Loss: {epoch_loss:.6f} | "
                f"Accuracy: "
                f"{epoch_accuracy * 100:.4f}% | "
                f"F1: {epoch_f1:.6f} | "
                f"Time: {epoch_seconds:.2f} seconds"
            ),
            log_file
        )

        # Match the learning-rate schedule used by the
        # original experiment.
        if epoch % 10 == 0:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] /= 10.0

            write_log(
                "Learning rate reduced to "
                f"{optimizer.param_groups[0]['lr']}",
                log_file
            )

    history_file = (
        output_folder / "training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_file,
        index=False
    )

    checkpoint_file = (
        output_folder
        / f"raw_arnn_epoch_{args.epochs}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "input_channels": 18,
            "segment_samples": 1024,
            "threshold": 0.5,
            "total_training_seconds": (
                total_training_seconds
            ),
        },
        checkpoint_file
    )

    configuration = {
        "architecture": "Original ARNN",
        "architecture_changed": False,
        "training_cases": [
            f"chb{i:02d}" for i in range(1, 10)
        ],
        "testing_cases": [
            "chb10",
            "chb11",
            "chb12"
        ],
        "patient_overlap": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "seed": args.seed,
        "input_shape": [1024, 18],
        "threshold": 0.5,
        "total_training_seconds": (
            total_training_seconds
        ),
    }

    with open(
        output_folder / "training_configuration.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2
        )

    write_log(
        "=" * 70,
        log_file
    )

    write_log(
        "TRAINING COMPLETED",
        log_file
    )

    write_log(
        f"Total training time: "
        f"{total_training_seconds:.2f} seconds",
        log_file
    )

    write_log(
        f"Checkpoint: {checkpoint_file}",
        log_file
    )

    write_log(
        f"History: {history_file}",
        log_file
    )

    write_log(
        "=" * 70,
        log_file
    )


if __name__ == "__main__":
    main()
