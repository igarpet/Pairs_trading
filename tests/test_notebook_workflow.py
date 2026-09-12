"""Execute every numbered notebook, each in a fresh Jupyter kernel."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from src.convergence_signal import simulate_unconditional_fou_paths

ORDER = [
    "01_Stock_Data.ipynb",
    "01_RF_Data.ipynb",
    "02_Pair_Selection.ipynb",
    "03_fractional_OU.ipynb",
    "03_pair_eligibility.ipynb",
    "04_convergence_signal.ipynb",
    "05_volatility_model.ipynb",
    "06_option_pricing.ipynb",
    "07_backtest.ipynb",
    "08_drawdown_diagnostics.ipynb",
    "09_systematic_risk_diagnostics.ipynb",
    "10_equilibrium_shift_diagnostics.ipynb",
    "11_static_convergence_calibration.ipynb",
    "12_alpha_validation.ipynb",
]


def test_every_module_in_separate_kernel(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    pytest.importorskip("ipykernel")

    def execute(notebook):
        if os.environ.get("NOTEBOOK_TEST_EXECUTOR") == "ipython":
            # Socket-free fallback for restricted runners; CI uses real kernels.
            target = repo / "executing.ipynb"
            nbformat.write(notebook, target)
            script = """
import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
shell=TerminalInteractiveShell.instance()
nb=nbformat.read('executing.ipynb',as_version=4)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code':
        result=shell.run_cell(cell.source,store_history=True)
        if not result.success:
            raise RuntimeError(f'Notebook cell {i} failed')
        cell.execution_count=shell.execution_count-1
nbformat.write(nb,'executing.ipynb')
"""
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            notebook.cells = nbformat.read(target, as_version=4).cells
        else:
            NotebookClient(
                notebook,
                timeout=180,
                kernel_name="python3",
                resources={"metadata": {"path": str(repo)}},
            ).execute()

    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "notebook_repo"
    repo.mkdir()
    for name in ["src"]:
        shutil.copytree(
            root / name, repo / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ORDER + ["requirements.txt", "METHODOLOGY.md"]:
        shutil.copyfile(root / name, repo / name)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = 4.5 + np.cumsum(np.random.default_rng(51).normal(0, 0.015, n))
    noise = simulate_unconditional_fou_paths(
        0, 0, 0.2, 0.015, 0.3, n - 1, n_paths=1, seed=32
    )[0]
    prices = pd.DataFrame({"A": np.exp(x + noise), "B": np.exp(x)}, index=idx)
    data = repo / "data/inputs"
    data.mkdir(parents=True)
    prices.to_parquet(data / "prices.parquet")
    pd.DataFrame({"risk_free_rate": 0.03}, index=idx).to_parquet(
        data / "risk_free_rates.parquet"
    )
    prices[["B"]].to_parquet(data / "sp500_prices.parquet")
    first = repo / ORDER[0]
    nb = nbformat.read(first, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace(
                "CONFIG = ResearchConfig()",
                "CONFIG = ResearchConfig(memory_window=10, structural_horizon=30, max_horizon_days=20, n_paths=100, n_placebos=2, bootstrap_replications=100, entry_z=.5)",
            )
    nbformat.write(nb, first)
    for name in ORDER:
        notebook = nbformat.read(repo / name, as_version=4)
        nbformat.validate(notebook)
        execute(notebook)
        nbformat.write(
            notebook, repo / name
        )  # Saving executed notebooks must not block later modules.
    out = repo / "data/processed/current"
    assert not (out / "manifest.json").exists()
    assert not (repo / "runs").exists()
    trades = pd.read_parquet(out / "trades.parquet")
    eq = pd.read_parquet(out / "equity_curve.parquet")
    assert len(trades) > 0
    from src.research_config import ResearchConfig

    assert eq.equity.iloc[0] == ResearchConfig().initial_capital
    assert eq.equity.iloc[-1] == pytest.approx(
        ResearchConfig().initial_capital + trades.pnl.sum()
    )
    assert (trades.entry_date > trades.signal_date).all()
    assert len(pd.read_parquet(out / "placebos.parquet")) == 2
    assert (out / "equilibrium_shift_metrics.parquet").exists()
    assert (out / "traded_forecast_calibration.parquet").exists()
    # Poison old output: a fresh Module 12 must recompute rather than reuse it.
    (out / "placebo_0000.json").write_text('{"sampled_pairs": ["stale"]}')
    last = nbformat.read(repo / ORDER[-1], as_version=4)
    execute(last)
    assert json.loads((out / "placebo_0000.json").read_text())["sampled_pairs"] != [
        "stale"
    ]
    # Notebook edits and repeated upstream execution need no new identity.
    first = nbformat.read(repo / ORDER[0], as_version=4)
    first.cells.append(nbformat.v4.new_markdown_cell("Edited notebook"))
    nbformat.write(first, repo / ORDER[0])
    execute(first)
    execute(nbformat.read(repo / ORDER[1], as_version=4))


def test_all_root_notebooks_are_valid():
    nbformat = pytest.importorskip("nbformat")
    root = Path(__file__).resolve().parents[1]
    for name in ORDER + ["00_START_HERE.ipynb"]:
        nb = nbformat.read(root / name, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert code
        # User-saved outputs are allowed; integration tests rerun source cells.
        assert all("subprocess.run" not in c.source for c in code)
        assert all(
            "NotebookSession" not in c.source and "RUN_NAME" not in c.source
            for c in code
        )
