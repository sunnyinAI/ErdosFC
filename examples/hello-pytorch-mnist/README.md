# Hello, Erdos FC — Federated MNIST

A complete, runnable Federated Averaging (FedAvg) demo. It trains a small CNN
across several **simulated client sites** that never share their raw data — only
model updates are exchanged and averaged on the server.

## Run it

From this directory:

```bash
pip install torch torchvision        # if you don't already have them
python run_simulation.py
```

That's it. No `pip install` of Erdos FC itself is required — the script adds the
repo root to `sys.path` automatically.

> If MNIST can't be downloaded (offline, no torchvision, etc.), the example
> automatically falls back to a small **synthetic** image dataset so it still
> runs end-to-end.

## What you'll see

Each round, every site trains locally for one epoch; the server averages their
weights and evaluates the new global model on a held-out test set:

```
[round 1/5] round=1, clients=4, test_acc=0.9012, test_loss=0.3380
[round 2/5] round=2, clients=4, test_acc=0.9447, test_loss=0.1902
...
```

The final global model is saved to `global_model.pt`.

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--clients` | `4` | number of simulated sites |
| `--rounds` | `5` | federated communication rounds |
| `--epochs` | `1` | local epochs per round |
| `--lr` | `0.01` | local SGD learning rate |
| `--batch-size` | `32` | local mini-batch size |
| `--dp-sigma` | `0.0` | Gaussian privacy noise on each client update (e.g. `0.01`) |

Turn on the privacy filter and watch the accuracy/robustness trade-off:

```bash
python run_simulation.py --dp-sigma 0.02
```

## How it maps to Erdos FC

| Concept | This example |
|---------|--------------|
| `PTTrainer` (Executor) | local CNN training on each site's shard |
| `Client` | `site-1 … site-N`, each with a private `Subset` |
| `FedAvgAggregator` | sample-count weighted averaging on the server |
| `FedAvg` (Controller) | the round loop: scatter → train → gather → aggregate |
| `GaussianPrivacyFilter` | optional noise on the client result path (`--dp-sigma`) |
| `Simulator` | runs the whole federation in one process |
