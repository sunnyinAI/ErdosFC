# Changelog

All notable changes to **Erdos Federated Computing** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-14

First public release. See the full notes in
[`docs/release_notes/efc_0.1.0.md`](docs/release_notes/efc_0.1.0.md).

### Added
- **Core APIs** — `Shareable` data-exchange object, `FLContext`, and base
  components (`Executor`, `Aggregator`, `Filter`, `Controller`).
- **FedAvg workflow** — a server-side controller that scatters the global model,
  gathers client updates, and aggregates them across configurable rounds.
- **Weighted FedAvg aggregator** — sample-count weighted averaging that is
  dtype-safe for both trainable weights and integer buffers.
- **Privacy filter** — `GaussianPrivacyFilter` providing L2 clipping plus
  Gaussian noise on the client result path.
- **Runtime** — `Server`, `Client`, and an in-process `Simulator` that runs a
  full federation on a single machine for development and research.
- **PyTorch Trainer** — `PTTrainer` executor that converts an ordinary PyTorch
  training loop into a federated client with minimal code.
- **Example** — `hello-pytorch-mnist`, a runnable end-to-end FedAvg demo with an
  automatic synthetic-data fallback when MNIST cannot be downloaded.

[0.1.0]: https://github.com/sunnyinAI/ErdosComputing/releases/tag/0.1.0
