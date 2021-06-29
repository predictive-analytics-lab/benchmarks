from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from time import monotonic
from typing import Optional
from typing_extensions import Final

import ethicml as em
from ethicml import vision as emvi
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb


class WandbMode(Enum):
    """Make W&B either log online, offline or not at all."""

    online = auto()
    offline = auto()
    disabled = auto()


class Dataset(Enum):
    """Which data to load."""

    celeba = auto()


@dataclass
class Config:
    data_dir: str
    dataset: Dataset
    img_size: int
    batch_size: int
    num_workers: int
    wandb: WandbMode
    gpu: int
    group: Optional[str] = None
    seed: int = 42
    pin_memory: bool = True
    non_blocking: bool = True


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config):
    # ===== settings =====
    device = torch.device(f"cuda:{cfg.gpu}")
    sens_attr: Final = "Male"
    target_attr: Final = "Smiling"
    print(OmegaConf.to_yaml(cfg, sort_keys=True, resolve=True))
    print("=" * 20)

    # ===== logging =====
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="benchmark",
        config=OmegaConf.to_container(cfg, enum_to_str=True, resolve=True),
        mode=cfg.wandb.name,
        group=cfg.group,
    )

    # ===== data =====
    if cfg.dataset is Dataset.celeba:
        dataset, base_dir = em.celeba(
            download_dir=cfg.data_dir, label=target_attr, sens_attr=sens_attr, download=False
        )
        assert dataset is not None, "could not load dataset"
        data_tup = dataset.load()
        data_tup, _ = em.train_test_split(data_tup, train_percentage=0.2, random_seed=cfg.seed)

        trafos = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        data = emvi.TorchImageDataset(data=data_tup, root=base_dir, transform=trafos)
    else:
        raise ValueError("unknown dataset")

    # ===== loading =====
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset=data,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    samples_per_second = np.zeros(len(loader))
    batches_per_second = np.zeros(len(loader))

    start = monotonic()
    start_total = monotonic()
    non_blocking = cfg.non_blocking
    for i, (x, _, _) in enumerate(tqdm(loader)):
        x = x.to(device, non_blocking=non_blocking)
        _ = x.mean()
        end = monotonic()

        samples_per_second_ = x.size(0) / (end - start)
        batches_per_second_ = 1 / (end - start)
        samples_per_second[i] = samples_per_second_
        batches_per_second[i] = batches_per_second_

        run.log({"samples_per_second": samples_per_second_}, step=i + 1)
        run.log({"batches_per_second": batches_per_second_}, step=i + 1)
        start = monotonic()

    total_time = monotonic() - start_total
    run.summary["total_samples_per_second"] = len(data) / total_time
    run.summary["total_batches_per_second"] = len(loader) / total_time

    for name, measurements in (("samples", samples_per_second), ("batches", batches_per_second)):
        for (agg_name, agg_func) in [
            ("mean", np.mean),
            ("std", np.std),
            ("median", np.median),
            ("min", np.min),
            ("max", np.max),
            ("25 percentile", lambda a: np.quantile(a, q=0.25)),
            ("75 percentile", lambda a: np.quantile(a, q=0.75)),
            ("IQR", lambda a: np.quantile(a, q=0.75) - np.quantile(a, q=0.25)),
        ]:
            agg = agg_func(measurements)
            print(f"{name}/{agg_name}: {agg:.4g}")
            run.summary[f"{name}/{agg_name}"] = agg

    run.finish()


if __name__ == "__main__":
    main()
