# Cluster-Aware Clustering

A research-oriented C++23 implementation of exact discrete-center solvers for
cluster-aware clustering objectives. The current focus is Top-$\ell$/L1
clustering, with a compact mixed-integer formulation as the practical baseline
and a custom SCIP constraint handler that explores dynamically separated
constraints as a path toward more general cluster norms.

The project is a correctness-first prototype rather than a general-purpose
clustering library. It includes independent solution verification, reproducible
JSON input and output, a command-line solver, and a deterministic primal warm
start.

## Problem

Given clients $I$, candidate centers $J$, distances $d_{ij}$, and a
required number of centers $k$, select exactly $k$ centers and assign every
client to an open center.

For Top-$\ell$/L1 clustering, each cluster is charged the sum of its
$\ell$ largest assignment distances, and the objective is the sum of these
cluster costs. This interpolates between objectives that emphasize only a
cluster's worst-served clients and the ordinary sum of assignment distances.

Two exact Top-$\ell$ formulations are implemented:

- **Compact:** uses the threshold identity
  $\operatorname{Top}_{\ell}(v) = \min_{t \geq 0}
  \ell t + \sum_i \max(v_i-t,0)$.
- **Dynamic:** represents each cluster cost with an exponential family of
  subset inequalities and separates violated inequalities through a custom
  SCIP constraint handler.

The compact solver also supports the standard L1/L1 objective.

The dynamic formulation is not currently expected to speed up Top-$\ell$/L1.
Its main purpose is architectural: a solver driven by a norm-specific
separation oracle may later generalize to broader, potentially arbitrary, inner
norms for which an appropriate oracle is available. That generalization is
future work; the current handler implements only Top-$\ell$ separation.

## Preliminary runtime status

The current Iris runs indicate that the compact formulation is the practical
choice at this stage:

| Iris instance | Compact formulation | Dynamic formulation |
| --- | --- | --- |
| 15 clients and centers | About 15 s | About 30 s |
| 30 clients and centers | About 4 min | About 2.5 h |

These are informal observations from the current development setup, not a
controlled benchmark or a claim of formulation speedup. Reproducible
measurements should record the hardware, build type, SCIP configuration,
termination condition, primal and dual bounds, and time to the first and best
incumbents.

## Implemented features

- compact exact models for L1/L1 and Top-$\ell$/L1;
- a dynamic exact Top-$\ell$/L1 model with a custom SCIP constraint handler;
- deterministic greedy initial incumbents shared by both formulations;
- versioned, strictly validated JSON instances and solutions;
- center-major rectangular distance matrices;
- an independent SCIP-free feasibility and objective checker;
- a formulation-agnostic SCIP wrapper with solve status and bound statistics;
- a single-instance command-line interface;
- unit, generated, exhaustive-small-instance, solver, and CLI tests;
- a dependency-free Python script for preparing two UCI Iris instances.

## Requirements

- CMake 3.20 or newer;
- a C++23 compiler (GCC 12 has been tested);
- [nlohmann/json](https://github.com/nlohmann/json) 3.11 or newer, available as
  a CMake package;
- [SCIP Optimization Suite](https://www.scipopt.org/) 10.0.x, available as a
  CMake package, for the exact solvers and CLI;
- Python 3.10 or newer for the data-preparation test and script.

SCIP is an external dependency and is not bundled with this repository.

## Build and test

Configure by pointing `SCIP_DIR` at the directory containing SCIP's
`scip-config.cmake`:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSCIP_DIR=/absolute/path/to/scip/lib/cmake/scip
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To build and test only the SCIP-independent core:

```sh
cmake -S . -B build-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLUSTER_AWARE_BUILD_SCIP=OFF
cmake --build build-core --parallel
ctest --test-dir build-core --output-on-failure
```

Pass `-DBUILD_TESTING=OFF` if Python and the test targets are not needed.

## End-to-end demo

The following commands start from the repository root and keep all generated
demo data under the ignored `build/` directory. Replace the `SCIP_DIR` value
with the directory containing your SCIP installation's `scip-config.cmake`.

```sh
# 1. Configure, build, and test.
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSCIP_DIR=/absolute/path/to/scip/lib/cmake/scip
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# 2. Download UCI Iris and create the 15- and 30-point instances.
./scripts/prepare_uci_iris.py \
  --output-dir build/demo/iris \
  --force

# 3. Solve the 15-point instance with the compact formulation.
./build/cluster-aware \
  --input build/demo/iris/iris-15.json \
  --output build/demo/iris-15-compact-solution.json \
  --formulation compact

# 4. Solve the same instance with dynamic separation.
./build/cluster-aware \
  --input build/demo/iris/iris-15.json \
  --output build/demo/iris-15-dynamic-solution.json \
  --formulation dynamic

# 5. Inspect the independently verified solutions.
python3 -m json.tool build/demo/iris-15-compact-solution.json
python3 -m json.tool build/demo/iris-15-dynamic-solution.json
```

On the current development setup, the 15-point preset takes about 15 seconds
with the compact formulation and 30 seconds with dynamic separation. The
generated `iris-30.json` is available for a larger compact-model run, which
takes about 4 minutes. The corresponding dynamic estimate is about 2.5 hours,
using the mean of two completed runs, so it is not part of this short demo.

## Command-line usage

Solve a Top-$\ell$/L1 instance with the compact formulation:

```sh
./build/cluster-aware \
  --input examples/top_l_l1.json \
  --output solution.json \
  --formulation compact
```

Use the dynamic formulation:

```sh
./build/cluster-aware \
  --input examples/top_l_l1.json \
  --output solution.json \
  --formulation dynamic
```

For L1/L1 instances, omit `--formulation` or select `compact`. The dynamic
formulation currently supports only Top-$\ell$/L1.

Before solving, the CLI constructs the same deterministic greedy incumbent for
both formulations and submits its complete original-space encoding to SCIP.
Any returned incumbent is checked independently before its JSON file is
written.

The process exits with:

- `0` when an optimal solution was proven and written;
- `1` for input, parsing, I/O, or solver errors;
- `2` for invalid command-line arguments;
- `3` when no incumbent exists, or when an incumbent was written but has not
  been proven optimal.

Run `./build/cluster-aware --help` for the complete option summary.

## JSON instances

The files in `examples/` show the supported version-1 input format. Each
instance specifies:

- a stable instance identifier;
- the number of centers to open;
- inner- and outer-norm types and parameters;
- client and candidate-center counts;
- a dense center-major distance matrix.

The parser rejects unknown fields, inconsistent dimensions, unsupported norm
combinations, invalid values, and invalid cluster counts rather than silently
repairing input.

## Preparing the Iris examples

The preparation script downloads the corrected UCI Iris archive, verifies its
SHA-256 checksum and structure, and creates deterministic nested instances with
15 and 30 observations:

```sh
./scripts/prepare_uci_iris.py --output-dir instances/iris
```

Use `--standardize` to z-score the four features before distances are computed,
and `--force` to replace existing outputs. Generated `instances/` are
intentionally excluded from Git.

The source dataset is:

> Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C56C76>

UCI distributes Iris under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Repository layout

```text
examples/       Small hand-written JSON instances
include/core/   SCIP-independent data structures, norms, checker, and heuristic
include/io/     JSON interface
include/scip/   Public exact-solver interfaces
scripts/        Reproducible dataset preparation
src/CLI/        Command-line executable
src/core/       Core implementations
src/io/         JSON implementation
src/scip/       Compact model, dynamic handler, and SCIP adapter
tests/          Core, I/O, SCIP, CLI, and script tests
```

## Planned performance work

The current greedy incumbent is deliberately simple. Planned improvements
include structure-aware constructive assignments, relocation and center-swap
local search, and relaxation-informed primal heuristics. Further work will also
investigate cut prioritization, separation frequency, and other constraint
handler settings.

A small reproducible benchmark runner and explicit CLI limits for solve time or
node count are natural next additions. They would make formulation experiments
safer to run and make performance claims easier to reproduce.

## Current scope

- Centers are selected from a finite candidate set; continuous center placement
  is outside the current scope.
- Exact solver support is currently limited to L1/L1 and Top-$\ell$/L1.
- The dynamic constraint handler is implemented for correctness and has not yet
  been tuned across a comprehensive benchmark suite.
- General synthetic-instance generation, automated benchmarking, and the
  Ball-k-Median approximation track are not yet implemented.

## Research background

The implementation is motivated by the broader $(f,g)$-clustering framework:

- [Clustering to Minimize Cluster-Aware Norm Objectives](https://arxiv.org/abs/2410.24104)
- [A Broader View on Clustering under Cluster-Aware Norm Objectives](https://arxiv.org/abs/2512.06211)

## License

Copyright (C) 2026 Martin G. Herold.

This project is licensed under the GNU General Public License, version 3 only
(`GPL-3.0-only`). See [LICENSE](LICENSE) for the complete license text.
