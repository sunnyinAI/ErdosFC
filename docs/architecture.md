# Architecture

Erdos FC follows the **controller / worker** split used by mature federated
learning runtimes. The split keeps a clean separation between *orchestration*
(what happens each round, on the server) and *local computation* (how a site
trains, on the client).

## The round loop

A federated run is a loop of communication **rounds**. Each round:

1. **Scatter** — the `Controller` sends the current global model to every client
   as a `Shareable` carrying a `train` task.
2. **Local train** — each client's `Executor` loads those weights, trains on the
   site's private data, and returns updated weights plus metadata (sample count,
   loss).
3. **Filter** — on the way out, each client's `result_filters` may transform the
   update (e.g. add privacy noise). The server never sees the raw local weights.
4. **Gather + aggregate** — the server feeds every client result into the
   `Aggregator`, which produces the next global model.
5. **Evaluate / persist** — optionally score the new global model and save it.

```
 round r:

   global model w_r
        │  scatter (Shareable: params=w_r)
        ▼
   ┌─────────── each site i ───────────┐
   │  Executor.execute(train, w_r)     │   trains on private D_i
   │      → w_r^(i)                     │
   │  result_filters (e.g. DP noise)   │
   │      → ŵ_r^(i)                     │
   └───────────────┬───────────────────┘
                   │  gather
                   ▼
   Aggregator.accept(ŵ_r^(i))  for all i
   Aggregator.aggregate()  →  w_{r+1} = Σ_i (n_i / Σ n) · ŵ_r^(i)
```

The aggregation shown is **Federated Averaging** (McMahan et al., 2017): a
weighted mean of client weights, weighted by each site's number of samples `n_i`.

## Components and where they run

| Component    | Runs on | Responsibility |
|--------------|---------|----------------|
| `Controller` | server  | drives the round loop (`FedAvg`) |
| `Aggregator` | server  | combines client updates (`FedAvgAggregator`) |
| `Server`     | server  | holds the global model; `broadcast()` reaches clients |
| `Client`     | site    | binds an `Executor` + filter chains to a site name |
| `Executor`   | site    | local training (`PTTrainer`) |
| `Filter`     | path    | transforms a `Shareable` in flight (`GaussianPrivacyFilter`) |
| `Shareable`  | wire    | the message envelope: `params` + `meta` |
| `FLContext`  | both    | run-scoped shared state (round, current client, …) |
| `Simulator`  | local   | runs the whole federation in one process |

## The communication contract

The `Controller` never talks to clients directly — it calls a single primitive
on the server:

```python
results = server.broadcast(task_name, shareable, fl_ctx)
# -> list of (client_name, result_shareable)
```

In this reference runtime, `Simulator` + `Server` implement `broadcast` **in
process and sequentially**: each client gets an independent deep copy of the
payload, runs its task, and returns a result. Because the controller depends
only on this contract — not on *how* messages travel — the same `FedAvg`
controller and `PTTrainer` executor you debug in the simulator can run unchanged
on a networked transport (e.g. gRPC) that implements the same `broadcast`
semantics. That is the intended path from simulation to a real deployment.

## Design principles

- **Framework-agnostic core.** `erdos.apis`, the aggregator, the controller, and
  the runtime import no deep-learning library. They operate on a generic
  `params` mapping via tensor methods, so a TensorFlow or JAX executor can drop
  in beside `PTTrainer`.
- **Everything is swappable.** Aggregation, privacy, the workflow, and local
  training are independent extension points behind small abstract base classes.
- **Data never leaves the site.** Only `Shareable` payloads cross the boundary,
  and `result_filters` are the last code to touch an update before it does.
