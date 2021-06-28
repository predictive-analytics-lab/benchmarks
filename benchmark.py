from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from time import monotonic
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
    batch_size: int
    num_workers: int
    wandb: WandbMode


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config):
    # ===== settings =====
    device = torch.device("cuda:0")
    sens_attr: Final = "Male"
    target_attr: Final = "Smiling"

    # ===== logging =====
    run = wandb.init(
        entity="predictive-analytics-lab",
        project="benchmark",
        config=OmegaConf.to_container(cfg, enum_to_str=True, resolve=True),
        mode=cfg.wandb.name,
    )

    # ===== data =====
    if cfg.dataset is Dataset.celeba:
        dataset, base_dir = em.celeba(
            download_dir=cfg.data_dir,
            label=target_attr,
            sens_attr=sens_attr,
            download=False,
            check_integrity=True,
        )
        assert dataset is not None
        data_tup = dataset.load()

        data = emvi.TorchImageDataset(data=data_tup, root=base_dir)
    else:
        raise ValueError("unknown dataset")

    # ===== loading =====
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]] = DataLoader(
        dataset=data, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    batch_times = np.zeros(len(loader))

    for i, (x, s, y) in enumerate(loader):
        start = monotonic()
        x = x.to(device)
        _ = x.mean()
        end = monotonic()

        batch_time = end - start
        batch_times[i] = batch_time

        run.log({"batch_time": batch_time}, step=i + 1)

    for (name, func) in [
        ("mean", np.mean),
        ("std", np.std),
        ("median", np.median),
        ("min", np.min),
        ("max", np.max),
        ("25 percentile", lambda a: np.quantile(a, q=0.25)),
        ("75 percentile", lambda a: np.quantile(a, q=0.75)),
    ]:
        run.summary[f"aggregates/{name}"] = func(batch_times)

    run.finish()
