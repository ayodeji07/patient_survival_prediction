"""
patient_survival/cli.py
────────────────────────────────────────────────────────────────
Command-line interface for the patient survival predictor.

After `pip install -e .`, the `predict` command is available
globally in the terminal.

Commands
────────
  predict run       — run predictions on a CSV file
  predict train     — train all models on a dataset
  predict evaluate  — evaluate trained models
  predict info      — show model information

Usage examples
──────────────
  # Predict mortality risk for new patients
  predict run --input patients.csv --output predictions.csv

  # Predict with SHAP explanations
  predict run --input patients.csv --output results.csv --explain

  # Use a specific model
  predict run --input patients.csv --model random_forest

  # Train all models
  predict train --dataset whas500

  # Evaluate trained models on test set
  predict evaluate --dataset whas500
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app     = typer.Typer(
    name    = "predict",
    help    = "Patient Survival & Outcome Prediction CLI",
    add_completion = False,
)
console = Console()


# ── predict run ───────────────────────────────────────────────────

@app.command("run")
def run_predictions(
    input_path: Path = typer.Option(
        ..., "--input", "-i",
        help = "Path to input CSV with patient features.",
    ),
    output_path: Path = typer.Option(
        Path("predictions.csv"), "--output", "-o",
        help = "Path for prediction results CSV.",
    ),
    model: str = typer.Option(
        "xgboost", "--model", "-m",
        help = "Model to use: logistic_regression, random_forest, xgboost, lightgbm.",
    ),
    dataset: str = typer.Option(
        "whas500", "--dataset", "-d",
        help = "Dataset the model was trained on: whas500 or uci.",
    ),
    explain: bool = typer.Option(
        False, "--explain", "-e",
        help = "Include SHAP explanations in output.",
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t",
        help = "Classification threshold (0.0–1.0).",
    ),
) -> None:
    """Run mortality risk predictions on a CSV file of patients.

    The input CSV must contain the clinical feature columns matching
    the training dataset.  Run `predict info --dataset whas500` to
    see the required columns.

    \b
    Example:
        predict run --input new_patients.csv --output results.csv
        predict run --input patients.csv --model xgboost --explain
    """
    if not input_path.exists():
        rprint(f"[red]Error:[/red] Input file not found: {input_path}")
        raise typer.Exit(1)

    rprint(f"\n[bold blue]Patient Survival Predictor[/bold blue]")
    rprint(f"  Model   : [cyan]{model}[/cyan]")
    rprint(f"  Dataset : [cyan]{dataset}[/cyan]")
    rprint(f"  Input   : {input_path}")
    rprint(f"  Output  : {output_path}\n")

    with console.status("Loading model..."):
        from patient_survival.predictor import SurvivalPredictor
        predictor = SurvivalPredictor(model_name=model, dataset=dataset)

    import pandas as pd
    df = pd.read_csv(input_path)
    rprint(f"Loaded [bold]{len(df)}[/bold] patients from {input_path.name}")

    with console.status(f"Running predictions{'+ SHAP' if explain else ''}..."):
        if explain:
            results = predictor.explain(df)
            out_df  = pd.DataFrame(results["predictions"])

            # Print top global features
            rprint("\n[bold]Top features driving predictions:[/bold]")
            for feat in results["global_importance"][:5]:
                bar = "█" * int(feat["mean_abs_shap"] * 100)
                rprint(
                    f"  {feat['feature']:<20} {feat['mean_abs_shap']:.4f}  "
                    f"[blue]{bar}[/blue]"
                )
        else:
            out_df = predictor.predict(df, threshold=threshold)

    # Print summary table
    _print_risk_summary(out_df)

    out_df.to_csv(output_path, index=False)
    rprint(f"\n[green]✓[/green] Predictions saved to [bold]{output_path}[/bold]")


# ── predict train ─────────────────────────────────────────────────

@app.command("train")
def train_models(
    dataset: str = typer.Option(
        "whas500", "--dataset", "-d",
        help = "Dataset to train on: whas500 or uci.",
    ),
    no_save: bool = typer.Option(
        False, "--no-save",
        help = "Do not save trained models to disk.",
    ),
) -> None:
    """Train all four ML models on the specified dataset.

    Downloads the dataset if not already present, runs
    preprocessing, trains all models with cross-validation,
    and saves results to data/results/.

    \b
    Example:
        predict train --dataset whas500
        predict train --dataset uci
    """
    rprint(f"\n[bold blue]Training ML Pipeline[/bold blue]")
    rprint(f"  Dataset : [cyan]{dataset}[/cyan]\n")

    from src.data.extract import load_whas500, load_uci_heart
    from src.data.transform import prepare_whas500_for_ml, prepare_uci_for_ml
    from src.models.train import train_all_models
    from src.utils.config import Paths
    import pickle

    Paths.ensure_all()

    with console.status("Loading and preprocessing data..."):
        if dataset == "whas500":
            raw_df = load_whas500()
            X_train, X_test, y_train, y_test, scaler = prepare_whas500_for_ml(raw_df)
        else:
            raw_df = load_uci_heart()
            X_train, X_test, y_train, y_test, scaler = prepare_uci_for_ml(raw_df)

    # Save scaler for inference
    scaler_path = Paths.models_dir / f"{dataset}_scaler.pkl"
    Paths.models_dir.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    rprint(
        f"Data ready: [bold]{len(X_train)}[/bold] train, "
        f"[bold]{len(X_test)}[/bold] test"
    )

    with console.status("Training models (this may take a few minutes)..."):
        results = train_all_models(
            X_train, X_test, y_train, y_test,
            dataset=dataset, save_models=not no_save,
        )

    _print_model_comparison(results)
    rprint(f"\n[green]✓[/green] Training complete.")


# ── predict evaluate ──────────────────────────────────────────────

@app.command("evaluate")
def evaluate_models(
    dataset: str = typer.Option("whas500", "--dataset", "-d"),
) -> None:
    """Evaluate all trained models and print a comparison table.

    \b
    Example:
        predict evaluate --dataset whas500
    """
    import json
    from src.utils.config import Paths

    results_path = Paths.results / f"ml_results_{dataset}.json"
    if not results_path.exists():
        rprint(f"[red]No results found.[/red] Run: predict train --dataset {dataset}")
        raise typer.Exit(1)

    results = json.loads(results_path.read_text())
    _print_model_comparison(results)


# ── predict info ──────────────────────────────────────────────────

@app.command("info")
def show_info(
    dataset: str = typer.Option("whas500", "--dataset", "-d"),
) -> None:
    """Show required feature columns for a dataset.

    \b
    Example:
        predict info --dataset whas500
        predict info --dataset uci
    """
    from src.utils.constants import WHAS500_FEATURE_META, UCI_FEATURE_META

    if dataset == "whas500":
        meta = WHAS500_FEATURE_META
        title = "WHAS500 Feature Reference"
    else:
        meta = UCI_FEATURE_META
        title = "UCI Heart Disease Feature Reference"

    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Column name", style="cyan")
    table.add_column("Display name")
    table.add_column("Unit")
    table.add_column("Normal range")

    for col, info in meta.items():
        table.add_row(
            col,
            info["display"],
            info.get("unit", ""),
            info.get("normal") or "—",
        )

    console.print(table)


# ── Helpers ───────────────────────────────────────────────────────

def _print_risk_summary(df) -> None:
    """Print a risk level distribution summary."""
    table = Table(title="Prediction Summary", header_style="bold")
    table.add_column("Risk Level", style="bold")
    table.add_column("N Patients", justify="right")
    table.add_column("Percentage", justify="right")

    total = len(df)
    colours = {"Low": "green", "Moderate": "yellow", "High": "red"}

    for level in ["Low", "Moderate", "High"]:
        n   = (df["risk_level"] == level).sum()
        pct = n / total * 100
        col = colours.get(level, "white")
        table.add_row(
            f"[{col}]{level}[/{col}]",
            str(n),
            f"{pct:.1f}%",
        )

    table.add_row("[bold]Total[/bold]", str(total), "100%")
    console.print(table)


def _print_model_comparison(results: dict) -> None:
    """Print a formatted model comparison table."""
    table = Table(title="Model Comparison — Test Set", header_style="bold blue")
    table.add_column("Model",    style="cyan", width=25)
    table.add_column("AUC-ROC",  justify="right")
    table.add_column("Avg Prec", justify="right")
    table.add_column("F1",       justify="right")
    table.add_column("Recall",   justify="right")
    table.add_column("Train (s)", justify="right")

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("test_metrics", {}).get("roc_auc", 0),
        reverse=True,
    )

    for name, res in sorted_results:
        tm = res.get("test_metrics", {})
        table.add_row(
            name,
            f"{tm.get('roc_auc', 0):.4f}",
            f"{tm.get('average_precision', 0):.4f}",
            f"{tm.get('f1', 0):.4f}",
            f"{tm.get('recall', 0):.4f}",
            f"{res.get('training_time_s', 0):.1f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
