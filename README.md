<h1 align="center">Erdos Federated Computing</h1>

<p align="center">
  <em>Many collaborators, one model, no shared data.</em><br>
  A lightweight, extensible runtime for federated learning.
</p>

<p align="center">
  <a href="docs/release_notes/efc_0.1.0.md"><img alt="version" src="https://img.shields.io/badge/release-0.1.0-blue"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

---

## This repository contains two Erdos projects

| Project | What it is | Where |
|---------|------------|-------|
| **Erdos Federated Computing** (`erdos-fc`) | A lightweight runtime for **federated learning** — many sites train one model without sharing data. | [`erdos/`](erdos/) (this README) |
| **Erdos Platform** (`erdos-platform`) | An **enterprise agentic AI** platform — composable agent cards, multi-agent orchestration, human-in-the-loop safety, continual learning, and real-time observability, in five layers. | [`platform/`](platform/) · [platform README](platform/README.md) |

The **Erdos Platform** site is published from [`platform/web/`](platform/web/) to GitHub Pages
(`https://sunnyinAI.github.io/ErdosComputing/`). Try the agent platform in one command:

```bash
pip install -e ./platform
erdos-platform demo        # runs a governed, audited multi-agent pipeline offline
```

---

## Why "Erdos"?

Paul Erdős published with more than 500 co-authors — his entire body of work was
built on **collaboration**, captured today by the famous *Erdős number*.
Federated learning is collaboration of exactly that kind: many parties improve a
shared model together, **without anyone handing over their private data**. Erdos
FC is a small, readable runtime that makes that pattern easy to build and study.

## What it is

Erdos FC trains machine-learning models across multiple **sites** (devices,
hospitals, banks, phones) that keep their data local. Each round, sites train on
their own data and send only model **updates** to a server, which **aggregates**
them into a new global model. The framework gives you clean, swappable pieces for
each part of that loop:

- a **controller/worker** programming model (server-side workflow + client-side executors),
- pluggable **aggregation** strategies (FedAvg out of the box),
- **filters** on the message path for privacy/compression,
- a single-machine **simulator** so you can develop a full federation in one process.

It is deliberately compact and dependency-light — the core imports no deep-learning
library at all — so it is easy to read, extend, and use as a research scaffold.

## Install

```bash
git clone https://github.com/sunnyinAI/ErdosComputing.git
cd ErdosComputing
pip install -e .            # core (numpy only)
pip install -e ".[torch]"   # + PyTorch, to run the examples
```

## Quickstart

Run the bundled end-to-end FedAvg demo (trains a CNN across 4 simulated sites):

```bash
cd examples/hello-pytorch-mnist
python run_simulation.py
```

…or wire a federation yourself in a few lines:

```python
import erdos
from erdos import Client, FedAvg, FedAvgAggregator, PTTrainer, Simulator

# one trainer (Executor) per site, each with its own private dataset
clients = [
    Client(f"site-{i}", PTTrainer(model=make_model(), dataset=shard_i, epochs=1))
    for i, shard_i in enumerate(client_shards)
]

controller = FedAvg(
    num_rounds=5,
    initial_params=initial_weights,        # dict[str, tensor]
    aggregator=FedAvgAggregator(),
    evaluate_fn=evaluate_global_model,     # optional
)

Simulator(controller=controller, clients=clients).run()
```

Add privacy with one extra object on the client's result path:

```python
from erdos import GaussianPrivacyFilter
Client("site-1", trainer, result_filters=[GaussianPrivacyFilter(sigma=0.01)])
```

## Architecture at a glance

```
                    ┌──────────────────────────────────────────┐
                    │                 SERVER                     │
                    │   FedAvg Controller  ──►  FedAvgAggregator │
                    └───────▲───────────────────────┬───────────┘
        scatter global model│                       │ aggregate updates
                            │                       ▼
        ┌───────────────────┴───────────────────────────────────┐
        │                   │                   │                │
   ┌────┴─────┐       ┌─────┴────┐        ┌─────┴────┐     ┌─────┴────┐
   │  site-1  │       │  site-2  │        │  site-3  │ ... │  site-N  │
   │ PTTrainer│       │ PTTrainer│        │ PTTrainer│     │ PTTrainer│
   │ + Filter │       │ + Filter │        │ + Filter │     │ + Filter │
   └──────────┘       └──────────┘        └──────────┘     └──────────┘
      local data         local data          local data       local data
```

Read more in [`docs/architecture.md`](docs/architecture.md) and the concept
reference in [`docs/concepts.md`](docs/concepts.md).

## Project layout

```
erdos/                 the runtime
  apis/                Shareable, FLContext, base components
  aggregators/         FedAvgAggregator
  filters/             GaussianPrivacyFilter
  workflows/           FedAvg controller
  executors/           PTTrainer (PyTorch)
  server.py  client.py  simulator.py
examples/
  hello-pytorch-mnist/ runnable FedAvg demo
docs/                  architecture, concepts, release notes
web/                   project landing page
```

## Roadmap

- Additional aggregators: FedProx, FedAdam/FedOpt, scaffold-style corrections.
- Real network transport (gRPC) behind the existing `Server.broadcast` contract.
- Secure aggregation and a moments-accountant DP path.
- Provisioning: TLS certificates and per-site "startup kits" for cross-org runs.
- TensorFlow / JAX executors alongside `PTTrainer`.

## Citation

```bibtex
@software{gupta_erdos_fc_2026,
  author  = {Sunny Gupta},
  title   = {Erdos Federated Computing},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/sunnyinAI/ErdosComputing}
}
```

## Author

**Sunny Gupta** — PhD Scholar (Machine Learning), IIT Bombay.
Federated learning · distributed deep learning · privacy-preserving AI.
[Portfolio](https://sunnyinai.github.io) ·
[Google Scholar](https://scholar.google.com/citations?user=-I-B6DgAAAAJ&hl=en) ·
[LinkedIn](https://linkedin.com/in/isunnyi)

## License

Apache License 2.0 — see [LICENSE](LICENSE). © 2026 Sunny Gupta.

> Not affiliated with or endorsed by NVIDIA. Erdos FC is an independent,
> educational federated-learning runtime inspired by the design patterns common
> to modern FL frameworks.
