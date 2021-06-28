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
        data_tup, _ = em.train_test_split(data_tup, train_percentage=0.2, random_seed=0)

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
        dataset=data, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    samples_per_second = np.zeros(len(loader))

    start = monotonic()
    for i, (x, _, _) in enumerate(tqdm(loader)):
        x = x.to(device)
        _ = x.mean()
        end = monotonic()

        per_second = x.size(0) / (end - start)
        samples_per_second[i] = per_second

        run.log({"samples_per_second": per_second}, step=i + 1)
        start = monotonic()

    for (name, func) in [
        ("mean", np.mean),
        ("std", np.std),
        ("median", np.median),
        ("min", np.min),
        ("max", np.max),
        ("25 percentile", lambda a: np.quantile(a, q=0.25)),
        ("75 percentile", lambda a: np.quantile(a, q=0.75)),
    ]:
        agg = func(samples_per_second)
        print(f"aggregates/{name}: {agg:.4g}")
        run.summary[f"aggregates/{name}"] = agg

    run.finish()


if __name__ == "__main__":
    main()
