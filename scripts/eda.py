import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "fare_price"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_missing_plot(df: pd.DataFrame, out_dir: Path) -> None:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    missing_pct.plot(kind="bar", color="#2a9d8f")
    plt.title("Missing Value Percentage by Feature")
    plt.ylabel("Missing (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close()


def save_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    df[TARGET].plot(kind="hist", bins=40, color="#457b9d", edgecolor="black")
    plt.title("Fare Price Distribution")
    plt.xlabel("Fare Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "fare_distribution.png", dpi=150)
    plt.close()


def save_corr_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    im = plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_feature_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    features = [
        "demand_index",
        "seasonality_index",
        "days_to_departure",
        "fuel_cost_index",
        "competitor_price",
        "load_factor",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for i, col in enumerate(features):
        axes[i].scatter(df[col], df[TARGET], alpha=0.25, s=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(TARGET)
        axes[i].set_title(f"{col} vs {TARGET}")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_vs_fare_scatter.png", dpi=150)
    plt.close()


def save_categorical_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    cat_cols = ["origin", "destination", "cabin_class"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, cat_cols):
        groups = [g[TARGET].values for _, g in df.groupby(col)]
        labels = [k for k, _ in df.groupby(col)]
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_title(f"{TARGET} by {col}")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "categorical_boxplots.png", dpi=150)
    plt.close()


def summarize(df: pd.DataFrame) -> str:
    num_df = df.select_dtypes(include=["number"])
    corr_target = (
        num_df.drop(columns=[TARGET])
        .corrwith(num_df[TARGET])
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))).sum()

    cabin_stats = (
        df.groupby("cabin_class")[TARGET]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean", ascending=False)
    )

    route_stats = (
        df.groupby(["origin", "destination"])[TARGET]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
        .head(10)
    )

    lines = []
    lines.append("# Airfare EDA Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Rows: {df.shape[0]}")
    lines.append(f"- Columns: {df.shape[1]}")
    lines.append(f"- Target: `{TARGET}`")
    lines.append("")
    lines.append("## Data Quality")
    missing = (df.isna().sum()).sort_values(ascending=False)
    for col, cnt in missing.items():
        lines.append(f"- {col}: {cnt} missing")
    lines.append("")
    lines.append("## Numeric Summary")
    lines.append("```text")
    lines.append(num_df.describe().round(3).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Top Correlations With Fare Price")
    for col, val in corr_target.items():
        lines.append(f"- {col}: {val:.3f}")
    lines.append("")
    lines.append("## Outlier Counts (IQR Rule)")
    for col, cnt in outlier_counts.sort_values(ascending=False).items():
        lines.append(f"- {col}: {int(cnt)}")
    lines.append("")
    lines.append("## Fare by Cabin Class")
    lines.append("```text")
    lines.append(cabin_stats.round(2).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Top 10 Expensive Routes")
    lines.append("```text")
    lines.append(route_stats.round(2).to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Generated Charts")
    lines.append("- `missing_values.png`")
    lines.append("- `fare_distribution.png`")
    lines.append("- `correlation_heatmap.png`")
    lines.append("- `feature_vs_fare_scatter.png`")
    lines.append("- `categorical_boxplots.png`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for airfare dataset.")
    parser.add_argument("--data", default="data/airfare_sample.csv", type=str)
    parser.add_argument("--out-dir", default="reports/eda", type=str)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(data_path)

    save_missing_plot(df, out_dir)
    save_target_distribution(df, out_dir)
    save_corr_heatmap(df, out_dir)
    save_feature_scatter(df, out_dir)
    save_categorical_boxplots(df, out_dir)

    report = summarize(df)
    report_path = out_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"EDA complete. Report: {report_path}")


if __name__ == "__main__":
    main()
