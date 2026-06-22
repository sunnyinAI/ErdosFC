# Concepts

A short reference for the core abstractions. Every component subclasses
`FLComponent` (which just provides a named `logger`).

## Shareable

The single message envelope exchanged between server and clients. It is a
`dict` subclass with two reserved sections:

- `params` — the model payload (a `state_dict`-style `{name: tensor}` mapping).
- `meta` — metadata (`num_samples`, `train_loss`, `current_round`, …).

```python
from erdos_fl import Shareable
s = Shareable(params=state_dict, meta={"num_samples": 1280})
s.params                 # the payload
s.get_meta_prop("num_samples", 1)
s.set_meta_prop("client", "site-1")
```

Because it is a plain dict, a `Shareable` is trivially serializable for a real
transport.

## FLContext

A run-scoped scratchpad passed to every component so they can share state
without being coupled. Well-known keys: `current_round`, `num_rounds`,
`current_client`.

```python
fl_ctx.get_prop(FLContext.CURRENT_ROUND)
```

## Executor  *(client-side)*

Local computation for a task. Implement `execute`:

```python
class MyTrainer(Executor):
    def execute(self, task_name, shareable, fl_ctx) -> Shareable:
        load(shareable.params); train(); 
        return Shareable(params=weights, meta={"num_samples": n})
```

`PTTrainer` is the bundled PyTorch implementation — it turns an ordinary
training loop into a federated client.

## Aggregator  *(server-side)*

Combines client results into one global update via `reset → accept* → aggregate`:

```python
agg.reset(fl_ctx)
for _, result in results: agg.accept(result, fl_ctx)
new_global = agg.aggregate(fl_ctx).params
```

`FedAvgAggregator` does sample-count weighted averaging and is dtype-safe (it
averages in float and restores each tensor's original dtype, so integer buffers
survive).

## Filter  *(message path)*

Transforms a `Shareable` in flight. A `Client` has two chains:

- `task_filters` — inbound, before `execute` (e.g. decompress/decrypt).
- `result_filters` — outbound, after `execute` (e.g. privacy noise).

`GaussianPrivacyFilter` provides optional L2 clipping + additive Gaussian noise
on the result path. (It is illustrative, not a calibrated `(ε, δ)` guarantee.)

## Controller  *(server-side workflow)*

Defines what happens each round. `FedAvg` implements the scatter → gather →
aggregate loop for a fixed number of rounds and exposes a `history` list of
per-round metrics. It accepts optional `evaluate_fn` and `persist_fn` callbacks.

## Server / Client / Simulator  *(runtime)*

- `Server` holds the global model and exposes `broadcast(task, shareable, ctx)`.
- `Client` pairs a site `name` with an `Executor` and its filter chains.
- `Simulator` wires a controller + clients (+ server) and runs the federation in
  one process — the fastest way to develop and debug before deployment.

```python
Simulator(controller=FedAvg(...), clients=[Client(...), ...]).run()
```
