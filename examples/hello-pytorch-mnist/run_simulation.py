# Copyright 2026 Sunny Gupta
# Licensed under the Apache License, Version 2.0 (the "License").
"""Hello, Erdos FC — a runnable end-to-end FedAvg demo.

Trains a small CNN across several simulated client sites using Federated
Averaging, evaluating the global model on a held-out test set after every round.

Run it::

    python run_simulation.py                 # 4 clients, 5 rounds
    python run_simulation.py --clients 8 --rounds 10
    python run_simulation.py --dp-sigma 0.01 # turn on the Gaussian privacy filter
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the `erdos` package importable when running this script directly,
# without `pip install`. (Repo root is two levels up from this file.)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

import erdos
from erdos import Client, FedAvg, FedAvgAggregator, GaussianPrivacyFilter, PTTrainer, Simulator

# Local modules (same directory as this script).
from data import load_data
from model import SmallCNN


def make_evaluator(test_dataset, device, batch_size: int = 256):
    """Build an evaluate_fn(global_params, round) -> {test_acc, test_loss}."""
    model = SmallCNN().to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    def evaluate(global_params, _round):
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss_sum += float(loss_fn(outputs, targets).item())
                correct += int((outputs.argmax(1) == targets).sum().item())
                total += targets.size(0)
        return {"test_acc": correct / total, "test_loss": loss_sum / total}

    return evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Erdos FC — federated MNIST demo")
    p.add_argument("--clients", type=int, default=4, help="number of client sites")
    p.add_argument("--rounds", type=int, default=5, help="federated rounds")
    p.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    p.add_argument("--lr", type=float, default=0.01, help="local SGD learning rate")
    p.add_argument("--batch-size", type=int, default=32, help="local batch size")
    p.add_argument(
        "--dp-sigma",
        type=float,
        default=0.0,
        help="std of Gaussian privacy noise on client updates (0 = off)",
    )
    p.add_argument("--data-root", type=str, default=str(Path(__file__).parent / "data"))
    p.add_argument("--save", type=str, default=str(Path(__file__).parent / "global_model.pt"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-16s | %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_datasets, test_dataset, source = load_data(
        args.clients, args.data_root, seed=args.seed
    )
    print(
        f"\nErdos Federated Computing v{erdos.__version__}\n"
        f"  data source : {source}\n"
        f"  device      : {device}\n"
        f"  clients     : {args.clients}  (sizes: {[len(d) for d in client_datasets]})\n"
        f"  test set    : {len(test_dataset)} samples\n"
        f"  privacy     : {'sigma=' + str(args.dp_sigma) if args.dp_sigma > 0 else 'off'}\n"
    )

    # Shared starting point: one model's initial weights, broadcast to all sites.
    initial_params = {
        k: v.detach().cpu().clone() for k, v in SmallCNN().state_dict().items()
    }

    # Build one client per site, each with its own private data shard.
    clients = []
    for i, shard in enumerate(client_datasets):
        trainer = PTTrainer(
            model=SmallCNN(),
            dataset=shard,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=str(device),
        )
        result_filters = []
        if args.dp_sigma > 0:
            result_filters = [GaussianPrivacyFilter(sigma=args.dp_sigma, seed=args.seed + i)]
        clients.append(Client(name=f"site-{i + 1}", executor=trainer, result_filters=result_filters))

    evaluate = make_evaluator(test_dataset, device)

    def persist(params, _round):
        torch.save(params, args.save)

    controller = FedAvg(
        num_rounds=args.rounds,
        initial_params=initial_params,
        aggregator=FedAvgAggregator(),
        evaluate_fn=evaluate,
        persist_fn=persist,
    )

    Simulator(controller=controller, clients=clients).run()

    # Summary
    print("\nRound-by-round test accuracy")
    print("  round |  acc   |  loss")
    print("  ------+--------+-------")
    for rec in controller.history:
        print(f"  {rec['round']:>5} | {rec.get('test_acc', float('nan')):.4f} | {rec.get('test_loss', float('nan')):.4f}")
    print(f"\nFinal global model saved to: {args.save}")


if __name__ == "__main__":
    main()
