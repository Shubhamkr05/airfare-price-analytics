import argparse
import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def build_notebook(data_path: str, report_path: str, charts_dir: str) -> dict:
    cells = []

    cells.append(
        md_cell(
            "# Airfare Price Prediction EDA\n\n"
            "This notebook performs exploratory data analysis on airfare pricing data.\n\n"
            f"- Data: `{data_path}`\n"
            f"- Report: `{report_path}`\n"
            f"- Charts directory: `{charts_dir}`"
        )
    )

    cells.append(
        code_cell(
            "import pandas as pd\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "\n"
            "DATA_PATH = Path('data/airfare_sample.csv')\n"
            "OUT_DIR = Path('reports/eda')\n"
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TARGET = 'fare_price'"
        )
    )

    cells.append(
        code_cell(
            "df = pd.read_csv(DATA_PATH)\n"
            "print('Shape:', df.shape)\n"
            "df.head()"
        )
    )

    cells.append(
        md_cell("## Data Quality")
    )
    cells.append(
        code_cell(
            "missing = df.isna().sum().sort_values(ascending=False)\n"
            "missing"
        )
    )
    cells.append(
        code_cell(
            "plt.figure(figsize=(10,5))\n"
            "(df.isna().mean() * 100).sort_values(ascending=False).plot(kind='bar', color='#2a9d8f')\n"
            "plt.title('Missing Value Percentage by Feature')\n"
            "plt.ylabel('Missing (%)')\n"
            "plt.tight_layout()\n"
            "plt.savefig(OUT_DIR / 'missing_values.png', dpi=150)\n"
            "plt.show()"
        )
    )

    cells.append(md_cell("## Target Distribution"))
    cells.append(
        code_cell(
            "plt.figure(figsize=(8,5))\n"
            "df[TARGET].plot(kind='hist', bins=40, color='#457b9d', edgecolor='black')\n"
            "plt.title('Fare Price Distribution')\n"
            "plt.xlabel('Fare Price')\n"
            "plt.ylabel('Count')\n"
            "plt.tight_layout()\n"
            "plt.savefig(OUT_DIR / 'fare_distribution.png', dpi=150)\n"
            "plt.show()"
        )
    )

    cells.append(md_cell("## Correlation Analysis"))
    cells.append(
        code_cell(
            "corr = df.select_dtypes(include=['number']).corr(numeric_only=True)\n"
            "plt.figure(figsize=(10,7))\n"
            "im = plt.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)\n"
            "plt.colorbar(im, fraction=0.046, pad=0.04)\n"
            "plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')\n"
            "plt.yticks(range(len(corr.index)), corr.index)\n"
            "plt.title('Correlation Heatmap (Numeric Features)')\n"
            "plt.tight_layout()\n"
            "plt.savefig(OUT_DIR / 'correlation_heatmap.png', dpi=150)\n"
            "plt.show()\n"
            "\n"
            "corr_target = df.select_dtypes(include=['number']).drop(columns=[TARGET]).corrwith(df[TARGET]).sort_values(key=lambda x: x.abs(), ascending=False)\n"
            "corr_target"
        )
    )

    cells.append(md_cell("## Feature vs Fare"))
    cells.append(
        code_cell(
            "features = ['demand_index','seasonality_index','days_to_departure','fuel_cost_index','competitor_price','load_factor']\n"
            "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
            "axes = axes.ravel()\n"
            "for i, col in enumerate(features):\n"
            "    axes[i].scatter(df[col], df[TARGET], alpha=0.25, s=12)\n"
            "    axes[i].set_xlabel(col)\n"
            "    axes[i].set_ylabel(TARGET)\n"
            "    axes[i].set_title(f'{col} vs {TARGET}')\n"
            "plt.tight_layout()\n"
            "plt.savefig(OUT_DIR / 'feature_vs_fare_scatter.png', dpi=150)\n"
            "plt.show()"
        )
    )

    cells.append(md_cell("## Categorical Impact"))
    cells.append(
        code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
            "for ax, col in zip(axes, ['origin', 'destination', 'cabin_class']):\n"
            "    groups = [g[TARGET].values for _, g in df.groupby(col)]\n"
            "    labels = [k for k, _ in df.groupby(col)]\n"
            "    ax.boxplot(groups, tick_labels=labels, showfliers=False)\n"
            "    ax.set_title(f'{TARGET} by {col}')\n"
            "    ax.tick_params(axis='x', rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.savefig(OUT_DIR / 'categorical_boxplots.png', dpi=150)\n"
            "plt.show()"
        )
    )

    cells.append(md_cell("## Route and Cabin Insights"))
    cells.append(
        code_cell(
            "cabin_stats = df.groupby('cabin_class')[TARGET].agg(['count','mean','median','min','max']).sort_values('mean', ascending=False)\n"
            "route_stats = df.groupby(['origin','destination'])[TARGET].agg(['count','mean','median']).sort_values('mean', ascending=False)\n"
            "print('Fare by cabin class:')\n"
            "display(cabin_stats.round(2))\n"
            "print('\\nTop expensive routes:')\n"
            "display(route_stats.head(10).round(2))"
        )
    )

    cells.append(md_cell("## Optional: Load generated markdown report"))
    cells.append(
        code_cell(
            "report_text = Path('reports/eda/eda_report.md').read_text(encoding='utf-8')\n"
            "print(report_text[:2000])"
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA notebook.")
    parser.add_argument("--data", default="data/airfare_sample.csv")
    parser.add_argument("--report", default="reports/eda/eda_report.md")
    parser.add_argument("--charts-dir", default="reports/eda")
    parser.add_argument("--output", default="reports/eda/EDA.ipynb")
    args = parser.parse_args()

    nb = build_notebook(args.data, args.report, args.charts_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Notebook created: {out}")


if __name__ == "__main__":
    main()
