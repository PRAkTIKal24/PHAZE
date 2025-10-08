# PHAZE Plotting Integration and Verification Plan

## 1. Introduction

This document outlines a plan to integrate a modular plotting framework into the PHAZE CLI. The primary goal is to generate plots for benchmarking various components of the PHAZE framework, focusing on performance metrics like time and memory consumption. These plots will be used in the upcoming research paper on PHAZE for low-latency ML inference in LHC triggers.

The plots will analyze the performance of:
- **Fingerprinting Algorithms:** Comparing different fingerprinting techniques.
- **ZKML Frameworks:** Comparing `ezkl`, `risc_zero`, and other backends for proof generation and verification.

The benchmarks will be run across different model complexities to provide a comprehensive overview of the trade-offs involved.

## 2. Proposed CLI Structure

A new `plot` subcommand will be added to the PHAZE CLI.

```bash
phaze plot <plot_type> [options]
```

### Plot Types:

- `fingerprinting`: Generates plots related to the fingerprinting stage.
- `zkml`: Generates plots related to the ZKML proof generation and verification stages.
- `full_phaze`: Generates plots for the end-to-end PHAZE pipeline.

### Options:

- `--model-complexities`: Path to a configuration file or a list of model complexities to test.
- `--num-runs`: Number of times to run each benchmark to get statistical data (default: 10).
- `--output-dir`: Directory to save the generated plots (default: `./plots/`).
- `--data-file`: Path to a file with pre-existing benchmark data to plot. If not provided, the benchmarks will be run.
- `--save-data`: Path to save the new benchmark data.

**Example Usage:**

```bash
# Generate and plot fingerprinting benchmarks
phaze plot fingerprinting --model-complexities low,medium,high --num-runs 10 --output-dir ./paper_plots/

# Generate and plot ZKML benchmarks
phaze plot zkml --model-complexities low,medium,high --num-runs 5 --output-dir ./paper_plots/

# Plot from existing data
phaze plot fingerprinting --data-file ./benchmark_data/fingerprinting.csv --output-dir ./paper_plots/
```

## 3. New Module Structure

A new `phaze/plotting` directory will be created to house the plotting logic. The existing benchmarking code will be refactored to be more modular and to save benchmark results.

```
phaze/
├── plotting/
│   ├── __init__.py
│   ├── main.py                 # Main entry point for the `phaze plot` command
│   ├── fingerprinting.py       # Logic for fingerprinting plots
│   ├── zkml.py                 # Logic for ZKML plots
│   └── utils.py                # Helper functions for plotting (e.g., styling)
├── benchmarking/
│   ├── __init__.py
│   ├── base_benchmark.py       # Base class for benchmarks
│   ├── fingerprinting.py       # Fingerprinting benchmark logic
│   ├── zkml.py                 # ZKML benchmark logic
│   └── results.py              # Saving and loading benchmark results
└── cli.py                      # Existing CLI file to be modified
```

### Key components:

- **`phaze/cli.py`**: This file (or equivalent) will be modified to add the new `plot` subcommand, delegating the logic to `phaze.plotting.main`.
- **`phaze.plotting.main.py`**: Parses CLI arguments for the `plot` command and calls the appropriate plotting function from `fingerprinting.py` or `zkml.py`.
- **`phaze.plotting.*.py`**: These files will use a plotting library like `matplotlib` or `seaborn` to generate plots from dataframes (likely using `pandas`).
- **`phaze.benchmarking.*.py`**: The existing benchmarking code will be refactored into this new directory. The benchmarks will be modified to return structured data (e.g., pandas DataFrames) that can be easily saved to CSV/JSON and later used for plotting.
- **`phaze.benchmarking.results.py`**: Will handle saving and loading of benchmark data, allowing for separation of data collection and plotting.

## 4. Data Collection and Storage

The refactored benchmarking modules will produce data in a structured format (pandas DataFrame), which will be saved to a CSV file.

An example `fingerprinting_benchmarks.csv` might look like this:

| model_complexity | algorithm | run | time_s | memory_mb |
|------------------|-----------|-----|--------|-----------|
| 1000             | poly_hash | 1   | 0.5    | 120       |
| 1000             | poly_hash | 2   | 0.52   | 121       |
| ...              | ...       | ... | ...    | ...       |
| 1000             | sha256    | 1   | 1.2    | 150       |
| ...              | ...       | ... | ...    | ...       |
| 5000             | poly_hash | 1   | 2.1    | 300       |
| ...              | ...       | ... | ...    | ...       |

This format will allow for easy filtering and aggregation when generating plots.

## 5. Plot Specifications

The following plots will be generated. All plots will include error bars or use violin plots/box plots to show the statistical spread from multiple runs.

### 5.1. Fingerprinting Plots

1.  **Time vs. Model Complexity:**
    -   **X-axis:** Model Complexity (e.g., number of parameters or FLOPs).
    -   **Y-axis:** Time taken (s).
    -   **Series:** One line for each fingerprinting algorithm.
2.  **Memory vs. Model Complexity:**
    -   **X-axis:** Model Complexity.
    -   **Y-axis:** Peak memory usage (MB).
    -   **Series:** One line for each fingerprinting algorithm.

### 5.2. ZKML Plots

1.  **Proof Generation Time vs. Model Complexity:**
    -   **X-axis:** Model Complexity.
    -   **Y-axis:** Time taken (s).
    -   **Series:** One line for each ZKML backend (`ezkl`, `risc_zero`, etc.).
2.  **Proof Generation Memory vs. Model Complexity:**
    -   **X-axis:** Model Complexity.
    -   **Y-axis:** Peak memory usage (MB).
    -   **Series:** One line for each ZKML backend.
3.  **Proof Verification Time vs. Model Complexity:**
    -   **X-axis:** Model Complexity.
    -   **Y-axis:** Time taken (s).
    -   **Series:** One line for each ZKML backend.
4.  **Proof Verification Memory vs. Model Complexity:**
    -   **X-axis:** Model Complexity.
    -   **Y-axis:** Peak memory usage (MB).
    -   **Series:** One line for each ZKML backend.

## 6. Verification Plan

The new functionality will be tested at multiple levels.

1.  **Unit Tests:**
    -   Test the benchmark data saving and loading functions in `phaze.benchmarking.results`.
    -   Test the plotting utility functions in `phaze.plotting.utils`.
    -   Test individual plotting functions with mock dataframes to ensure plots are generated without errors.

2.  **Integration Tests:**
    -   Add tests for the new CLI commands (`phaze plot ...`).
    -   These tests will run the plotting command with a minimal model and a small number of runs (`--num-runs=1`).
    -   The tests will check if plot files are created in the specified output directory.
    -   The tests will not check the plots for visual correctness, only for their existence.

3.  **Manual Verification:**
    -   Run the CLI commands with a wider range of options.
    -   Visually inspect the generated plots to ensure they are correct, legible, and follow the specifications (correct axes, legends, titles, statistical representation).
    -   Verify that the data in the saved CSV files matches the data represented in the plots.

## 7. Implementation Roadmap

1.  **Setup:** Create the new directory structure (`phaze/plotting`, `phaze/benchmarking`).
2.  **Refactor Benchmarking:**
    -   Move existing benchmarking logic into the new `phaze/benchmarking` directory.
    -   Modify each benchmark to return a pandas DataFrame.
    -   Implement the `phaze.benchmarking.results` module for saving/loading data.
3.  **Develop Plotting Module:**
    -   Implement the plotting functions in `phaze.plotting.fingerprinting` and `phaze.plotting.zkml`.
    -   Use `matplotlib`/`seaborn` for plotting.
4.  **Integrate with CLI:**
    -   Modify `phaze/cli.py` to add the `plot` subcommand.
    -   Implement `phaze.plotting.main` to handle the `plot` command arguments and call the correct functions.
5.  **Testing:**
    -   Write unit tests for the new modules.
    -   Write integration tests for the CLI.
6.  **Documentation:**
    -   Update `README.md` and any other relevant documentation to reflect the new `plot` command.
    -   Add docstrings to all new functions and modules.
7.  **Final Review:**
    -   Perform manual verification of the generated plots.
    -   Review the code for quality and adherence to project standards.
