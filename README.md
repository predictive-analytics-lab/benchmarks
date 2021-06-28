# Benchmarks

## Usage

Simple usage:

```
python benchmark.py +experiment=celeba_dataloading hydra/launcher=slurm_goedel
```

Multiple runs:

```
python benchmark.py -m +experiment=celeba_dataloading seed="range(0,5)" hydra/launcher=slurm_goedel hydra.launcher.cpus_per_task=10 num_workers=10 batch_size=64
```
