<h1 align="center">ErdosFL — Erdos Federated Learning</h1>

<p align="center">
  <em>Many collaborators, one outcome, no shared data.</em><br>
  An open foundation for collaborative, privacy-preserving AI — with two products:
  <b>ErdosFC</b> (federated computing) and <b>ErdosFAI</b> (federated agentic intelligence).
</p>

<p align="center">
  <a href="docs/release_notes/efc_0.1.0.md"><img alt="version" src="https://img.shields.io/badge/release-0.1.0-blue"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

---

## Two products under one roof

**ErdosFL** is the umbrella. It ships two products that share one principle —
collaborate on an outcome, never on raw data:

| Product | What it is | Where |
|---------|------------|-------|
| **ErdosFC** — *Erdos Federated Computing* | A lightweight runtime for **federated learning** — many sites train one model without sharing data. | code in [`erdos_fl/`](erdos_fl/) (documented below) |
| **ErdosFAI** — *Erdos Federated Agentic Intelligence* | An **enterprise agentic AI** platform — composable agent cards, multi-agent orchestration, human-in-the-loop safety, continual learning, and real-time observability, in five layers. | [`erdos-fai/`](erdos-fai/) · [README](erdos-fai/README.md) |

### Website

The unified ErdosFL site is published to GitHub Pages:

- **ErdosFL** (home) — <https://sunnyinai.github.io/ErdosFC/>
- **ErdosFC** — <https://sunnyinai.github.io/ErdosFC/fc/>
- **ErdosFAI** — <https://sunnyinai.github.io/ErdosFC/fai/>

Try the agent platform in one command:

```bash
pip install -e ./erdos-fai
erdos-fai demo        # runs a governed, audited multi-agent pipeline offline
```

---

## Why "Erdos"?

Paul Erdős published with more than 500 co-authors — his entire body of work was
built on **collaboration**, captured today by the famous *Erdős number*. Both
Erdos products are collaboration of exactly that kind: many parties (sites, or
agents) work toward a shared outcome **without anyone handing over their private
data**. ErdosFL is a small, readable home for that idea.

---

# ErdosFC — Erdos Federated Computing

> The federated-learning runtime. Its Python package is `erdos_fl`.

## What it is

ErdosFC trains machine-learning models across multiple **sites** (devices,
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
git clone https://github.com/sunnyinAI/ErdosFC.git
cd ErdosFC
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
import erdos_fl
from erdos_fl import Client, FedAvg, FedAvgAggregator, PTTrainer, Simulator

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
from erdos_fl import GaussianPrivacyFilter
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

## Repository layout

```
erdos_fl/              ErdosFC — the federated-learning runtime (package: erdos_fl)
  apis/                Shareable, FLContext, base components
  aggregators/         FedAvgAggregator
  filters/             GaussianPrivacyFilter
  workflows/           FedAvg controller
  executors/           PTTrainer (PyTorch)
  server.py  client.py  simulator.py
erdos-fai/             ErdosFAI — the agentic AI platform
  erdos_fai/           intelligence · orchestration · safety · learning · insights
  examples/  tests/  web/
web-home/              ErdosFL umbrella landing page (deployed to site root)
examples/              ErdosFC runnable FedAvg demo
docs/                  architecture, concepts, release notes
```

## Roadmap (ErdosFC)

- Additional aggregators: FedProx, FedAdam/FedOpt, scaffold-style corrections.
- Real network transport (gRPC) behind the existing `Server.broadcast` contract.
- Secure aggregation and a moments-accountant DP path.
- Provisioning: TLS certificates and per-site "startup kits" for cross-org runs.
- TensorFlow / JAX executors alongside `PTTrainer`.

## Citation

```bibtex
@software{gupta_erdos_fl_2026,
  author  = {Sunny Gupta},
  title   = {ErdosFL: Erdos Federated Learning},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/sunnyinAI/ErdosFC}
}
```

## Author

**Sunny Gupta** — PhD Scholar (Machine Learning), IIT Bombay.
Federated learning · distributed deep learning · privacy-preserving AI · agentic systems.
[Portfolio](https://sunnyinai.github.io) ·
[Google Scholar](https://scholar.google.com/citations?user=-I-B6DgAAAAJ&hl=en) ·
[LinkedIn](https://linkedin.com/in/isunnyi)

## License

Apache License 2.0 — see [LICENSE](LICENSE). © 2026 Sunny Gupta.

> ErdosFL (ErdosFC + ErdosFAI) is an independent, educational project. It is
> inspired by design patterns common to modern federated-learning and enterprise
> agent platforms, and is not affiliated with or endorsed by any of them.
