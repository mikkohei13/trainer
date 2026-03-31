"""Quality regression model training."""

import logging
import math
import multiprocessing
import random
from pathlib import Path

from PIL import Image

from trainer import db
from trainer.images import IMAGES_DIR

MODELS_DIR = Path(__file__).resolve().parent / "models"

QUALITY_BASE_MODEL = "resnet18"
QUALITY_TRAIN_EPOCHS = 40
QUALITY_PATIENCE = 8
QUALITY_LR = 3e-4
QUALITY_WEIGHT_DECAY = 1e-4
QUALITY_BATCH_SIZE = 32
QUALITY_IMG_SIZE = 224
BBOX_PADDING_FRACTION = 0.10


def start_quality_training_process(run_id: int, taxon: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_blocking_quality_train, args=(run_id, taxon), daemon=True)
    p.start()


def _blocking_quality_train(run_id: int, taxon: str) -> None:
    run_dir = MODELS_DIR / taxon / "quality" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    trainer_logger = logging.getLogger(f"quality_training.{run_id}")
    trainer_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    trainer_logger.addHandler(handler)

    trainer_logger.info("[quality run %s] starting for project '%s'", run_id, taxon)

    try:
        records = _collect_quality_records(taxon)
        if len(records) < 5:
            raise ValueError("Need at least 5 quality-rated images with bounding boxes for training")

        train_records, val_records = _split_records(records)
        trainer_logger.info(
            "[quality run %s] dataset size: train=%s, val=%s",
            run_id,
            len(train_records),
            len(val_records),
        )

        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        transform = Compose([
            Resize((QUALITY_IMG_SIZE, QUALITY_IMG_SIZE)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_ds = _QualityCropDataset(train_records, transform)
        val_ds = _QualityCropDataset(val_records, transform)
        train_loader = DataLoader(train_ds, batch_size=QUALITY_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=QUALITY_BATCH_SIZE, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid(),
        )
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=QUALITY_LR,
            weight_decay=QUALITY_WEIGHT_DECAY,
        )
        criterion = nn.MSELoss()

        best_rmse = None
        best_state = None
        stale_epochs = 0

        for epoch in range(1, QUALITY_TRAIN_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = x.size(0)
                train_loss_sum += float(loss.detach().cpu().item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(1, train_count)
            val_rmse = _evaluate_rmse(model, val_loader, device)
            trainer_logger.info(
                "[quality run %s] epoch=%s train_mse=%.6f val_rmse=%.6f",
                run_id,
                epoch,
                train_loss,
                val_rmse,
            )

            if best_rmse is None or val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= QUALITY_PATIENCE:
                    trainer_logger.info(
                        "[quality run %s] early stopping after %s stale epochs",
                        run_id,
                        stale_epochs,
                    )
                    break

        if best_state is None or best_rmse is None:
            raise RuntimeError("Training did not produce a model")

        best_path = run_dir / "best.pt"
        torch.save(
            {
                "model_name": QUALITY_BASE_MODEL,
                "img_size": QUALITY_IMG_SIZE,
                "padding_fraction": BBOX_PADDING_FRACTION,
                "state_dict": best_state,
            },
            best_path,
        )

        trainer_logger.info(
            "[quality run %s] done — val_rmse=%.6f, model=%s",
            run_id,
            best_rmse,
            best_path,
        )
        db.finish_quality_training_run(run_id, str(best_path), float(best_rmse), str(log_path))
    except Exception as exc:
        trainer_logger.exception("[quality run %s] failed: %s", run_id, exc)
        db.fail_quality_training_run(run_id, str(log_path))
    finally:
        handler.close()
        trainer_logger.removeHandler(handler)


class _QualityCropDataset:
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        src = IMAGES_DIR / rec["image_path"]
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            crop = _crop_box_with_padding(rgb, rec["box"], BBOX_PADDING_FRACTION)
        x = self.transform(crop)

        import torch

        y = torch.tensor(float(rec["quality"]), dtype=torch.float32)
        return x, y


def _crop_box_with_padding(img: Image.Image, box: dict, pad_frac: float) -> Image.Image:
    img_w, img_h = img.size
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])

    pad_x = w * pad_frac
    pad_y = h * pad_frac

    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(img_w, int(math.ceil(x + w + pad_x)))
    bottom = min(img_h, int(math.ceil(y + h + pad_y)))

    if right <= left or bottom <= top:
        return img.copy()
    return img.crop((left, top, right, bottom))


def _collect_quality_records(taxon: str) -> list[dict]:
    quality_map = db.get_image_quality_map(taxon)
    records = []
    for image_path, quality in quality_map.items():
        annotations = db.get_annotations(image_path)
        if not annotations["boxes"]:
            continue
        records.append({
            "image_path": image_path,
            "quality": float(quality),
            "box": annotations["boxes"][0],
        })
    records.sort(key=lambda r: r["image_path"])
    return records


def _split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    data = records[:]
    random.Random(42).shuffle(data)
    split = int(len(data) * 0.8)
    split = max(1, min(len(data) - 1, split))
    return data[:split], data[split:]


def _evaluate_rmse(model, dataloader, device) -> float:
    import torch

    model.eval()
    sq_error_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).squeeze(1)
            sq_error_sum += float(((pred - y) ** 2).sum().detach().cpu().item())
            count += x.size(0)
    mse = sq_error_sum / max(1, count)
    return math.sqrt(mse)
