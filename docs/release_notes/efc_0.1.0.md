# Erdos Federated Computing 0.1.0

*First public release — 14 June 2026*

Erdos FC 0.1.0 establishes the foundation of a lightweight, readable runtime for
federated learning: a clean controller/worker programming model, Federated
Averaging end to end, a privacy filter on the client path, and a single-machine
simulator that runs a whole federation in one process. It ships with a runnable
PyTorch demo so you can go from clone to a training federation in one command.

## Feature highlights

- **Controller / worker programming model.** A server-side `Controller` defines
  the federated workflow while client-side `Executor`s perform local training,
  mirroring the architecture of production FL runtimes but in a compact, legible
  codebase.

- **Federated Averaging.** The `FedAvg` controller drives the scatter → train →
  gather → aggregate loop over configurable rounds, and `FedAvgAggregator`
  performs sample-count-weighted averaging that is dtype-safe for both trainable
  weights and integer buffers.

- **Privacy on the message path.** `GaussianPrivacyFilter` provides optional
  global-L2 clipping plus additive Gaussian noise on the *client result path*,
  so the server never observes a site's raw update. Filters are a general
  extension point for compression and encryption too.

- **Single-machine simulator.** `Simulator` + `Server` execute the full
  federation in-process behind a single `broadcast()` contract, so the same
  controller and executors you debug locally can later run against a networked
  transport unchanged.

- **PyTorch "Client API".** `PTTrainer` converts an ordinary PyTorch training
  loop into a federated client with almost no changes — supply a model and a
  dataset and you have a site.

- **Framework-agnostic core.** The APIs, aggregator, workflow, and runtime
  import no deep-learning library, leaving room for TensorFlow/JAX executors
  beside `PTTrainer`.

- **Runnable example.** `hello-pytorch-mnist` trains a CNN across simulated
  sites with per-round test evaluation, and falls back to a synthetic dataset
  automatically when MNIST can't be downloaded — so it always runs end to end.

## What's included

- Core APIs: `Shareable`, `FLContext`, `FLComponent`, `Executor`, `Aggregator`,
  `Filter`, `Controller`, `TaskName`.
- Algorithms: `FedAvg` (controller), `FedAvgAggregator`.
- Privacy: `GaussianPrivacyFilter`.
- Runtime: `Server`, `Client`, `Simulator`.
- PyTorch: `PTTrainer`.
- Docs: architecture overview, concept reference, this release note.
- Example: `examples/hello-pytorch-mnist`.

## Known limitations

- The reference transport is in-process and sequential (by design, for
  development); there is no network layer yet.
- `GaussianPrivacyFilter` is illustrative and does **not** provide a calibrated
  `(epsilon, delta)` differential-privacy guarantee.
- Only FedAvg is included; FedProx/FedOpt and secure aggregation are on the
  roadmap.

## Looking ahead (0.2.0)

A gRPC transport behind the existing `broadcast` contract, additional
aggregators (FedProx, FedOpt), secure aggregation, and provisioning (TLS certs +
per-site startup kits) for real cross-organization runs.
