from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd


df_obs = pd.read_csv("out/observations.csv", sep=";")
df_perf = pd.read_csv("out/performance.csv", sep=";")


## TEXTUAL ANALYSIS

METRIC_ROWS = [("R2", "R²"), ("mean_error", "Mean error"), ("hit_rate", "Hit rate"), ("hit_rate±05", "Hit Rate ±0.5"), ("hit_rate±1", "Hit Rate ±1.0"),("std_diff", "std_diff")]

CHARACTERISTIC_ROWS = [("rated_movies", "Rated movies"), ("observations", "Observations"), ("followers", "Followers")]


def fmt_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def fmt_count(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(round(value)):,}"


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


def summarize(df: pd.DataFrame, column: str, weights: pd.Series | None = None) -> dict[str, float]:
    series = pd.to_numeric(df[column], errors="coerce")
    summary = {
        "mean": series.mean() if weights is None else weighted_mean(series, weights),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }
    return summary


def print_metric_block(stats_df: pd.DataFrame, weighted: bool = False) -> None:
    mean_header = "Weighted mean" if weighted else "Mean"
    print(f"{'':<20}{mean_header:>12}{'Median':>12}{'Std':>12}{'Min':>12}{'Max':>12}")
    for column, label in METRIC_ROWS:
        row = stats_df.loc[column]
        if column in {"hit_rate", "hit_rate±05", "hit_rate±1"}:
            mean = fmt_percent(row["mean"])
            median = fmt_percent(row["median"])
            std = fmt_percent(row["std"])
            min_ = fmt_percent(row["min"])
            max_ = fmt_percent(row["max"])
        else:
            mean = fmt_float(row["mean"])
            median = fmt_float(row["median"])
            std = fmt_float(row["std"])
            min_ = fmt_float(row["min"])
            max_ = fmt_float(row["max"])
        print(f"{label:<20}{mean:>12}{median:>12}{std:>12}{min_:>12}{max_:>12}")
    print()


def print_characteristics_block(stats_df: pd.DataFrame) -> None:
    print(f"{'':<20}{'Total':>12}{'Mean':>12}{'Median':>12}{'Std':>12}{'Min':>12}{'Max':>12}")
    for column, label in CHARACTERISTIC_ROWS:
        row = stats_df.loc[column]
        print(
            f"{label:<20}"
            f"{fmt_count(row['total']):>12}"
            f"{fmt_float(row['mean']):>12}"
            f"{fmt_float(row['median']):>12}"
            f"{fmt_float(row['std']):>12}"
            f"{fmt_count(row['min']):>12}"
            f"{fmt_count(row['max']):>12}"
        )


def build_metric_stats(df: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    stats = {}
    for column, _ in METRIC_ROWS:
        stats[column] = summarize(df, column, weights=weights)
    return pd.DataFrame.from_dict(stats, orient="index")


def build_characteristic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for column, _ in CHARACTERISTIC_ROWS:
        series = pd.to_numeric(df[column], errors="coerce")
        stats[column] = {
            "total": series.sum(),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
        }
    return pd.DataFrame.from_dict(stats, orient="index")


## VISUAL ANALYSIS

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def set_stata_like_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def format_integer_ticks(value: float, _pos: int) -> str:
    return f"{int(value):,}"


def format_metric_ticks(value: float, _pos: int) -> str:
    return f"{value:.2f}"


def plot_observations_r2_mean_error(df: pd.DataFrame) -> Path:
    data = df.copy()
    data["observations"] = pd.to_numeric(data["observations"], errors="coerce")
    data["R2"] = pd.to_numeric(data["R2"], errors="coerce")
    data["mean_error"] = pd.to_numeric(data["mean_error"], errors="coerce")
    data = data.dropna(subset=["observations", "R2", "mean_error"]).sort_values("observations")

    set_stata_like_style()
    fig, ax_left = plt.subplots(figsize=(8.6, 5.2))
    ax_right = ax_left.twinx()

    left_color = "0.28"
    right_color = "0.08"

    ax_right.scatter(
        data["observations"],
        data["mean_error"],
        marker="x",
        s=50,
        linewidths=1.0,
        color=left_color,
        label="Mean error",
        zorder=3,
    )
    ax_left.scatter(
        data["observations"],
        data["R2"],
        marker="o",
        s=36,
        facecolors=right_color,
        edgecolors="white",
        linewidths=0.4,
        label="R2",
        zorder=4,
    )

    ax_left.set_xlabel("Observations")
    ax_left.set_ylabel("Mean error", color=right_color)
    ax_right.set_ylabel("R2", color=left_color)
    ax_left.tick_params(axis="y", colors=right_color)
    ax_right.tick_params(axis="y", colors=left_color)
    ax_left.tick_params(axis="x", colors="0.2")

    ax_left.yaxis.set_major_formatter(FuncFormatter(format_metric_ticks))
    ax_right.yaxis.set_major_formatter(FuncFormatter(format_metric_ticks))
    ax_left.xaxis.set_major_formatter(FuncFormatter(format_integer_ticks))

    ax_left.grid(axis="y")
    ax_left.grid(axis="x", visible=False)
    ax_right.grid(False)

    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    ax_left.set_title("Performance by sample size", loc="left", pad=10, weight="bold")

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

    output_path = FIGURE_DIR / "performance_sample_size.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_std_diff_r2_hit_rate(df: pd.DataFrame) -> Path:
    data = df.copy()
    data["std_diff"] = pd.to_numeric(data["std_diff"], errors="coerce")
    data["R2"] = pd.to_numeric(data["R2"], errors="coerce")
    data["hit_rate"] = pd.to_numeric(data["hit_rate"], errors="coerce")
    data = data.dropna(subset=["std_diff", "R2", "hit_rate"]).sort_values("std_diff")

    set_stata_like_style()
    fig, ax_left = plt.subplots(figsize=(8.6, 5.2))
    ax_right = ax_left.twinx()

    left_color = "0.28"
    right_color = "0.08"

    ax_right.scatter(
        data["std_diff"],
        data["hit_rate"],
        marker="x",
        s=50,
        linewidths=1.0,
        color=left_color,
        label="Hit rate",
        zorder=3,
    )
    ax_left.scatter(
        data["std_diff"],
        data["R2"],
        marker="o",
        s=36,
        facecolors=right_color,
        edgecolors="white",
        linewidths=0.4,
        label="R2",
        zorder=4,
    )

    ax_left.set_xlabel("std_diff")
    ax_left.set_ylabel("R2", color=right_color)
    ax_right.set_ylabel("Hit rate", color=left_color)
    ax_left.tick_params(axis="y", colors=right_color)
    ax_right.tick_params(axis="y", colors=left_color)
    ax_left.tick_params(axis="x", colors="0.2")

    ax_left.yaxis.set_major_formatter(FuncFormatter(format_metric_ticks))
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}%"))
    ax_left.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.3f}"))

    ax_left.grid(axis="y")
    ax_left.grid(axis="x", visible=False)
    ax_right.grid(False)

    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    ax_left.set_title("Performance by standard deviation gap", loc="left", pad=10, weight="bold")

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

    output_path = FIGURE_DIR / "performance_std_diff.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_observations_hit_rates(df: pd.DataFrame) -> Path:
    data = df.copy()
    data["observations"] = pd.to_numeric(data["observations"], errors="coerce")
    data["hit_rate"] = pd.to_numeric(data["hit_rate"], errors="coerce")
    data["hit_rate±05"] = pd.to_numeric(data["hit_rate±05"], errors="coerce")
    data["hit_rate±1"] = pd.to_numeric(data["hit_rate±1"], errors="coerce")
    data = data.dropna(subset=["observations", "hit_rate", "hit_rate±05", "hit_rate±1"]).sort_values("observations")

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    ax.scatter(
        data["observations"],
        data["hit_rate"],
        marker="o",
        s=28,
        facecolors="0.10",
        edgecolors="white",
        linewidths=0.3,
        label="Hit rate",
        zorder=4,
    )
    ax.scatter(
        data["observations"],
        data["hit_rate±05"],
        marker="x",
        s=40,
        linewidths=0.8,
        color="0.35",
        label="Hit rate ±0.5",
        zorder=3,
    )
    ax.scatter(
        data["observations"],
        data["hit_rate±1"],
        marker="^",
        s=32,
        facecolors="0.55",
        edgecolors="white",
        linewidths=0.3,
        label="Hit rate ±1.0",
        zorder=2,
    )

    ax.set_xlabel("Observations")
    ax.set_ylabel("Hit rate")
    ax.tick_params(axis="x", colors="0.2")
    ax.tick_params(axis="y", colors="0.2")
    ax.xaxis.set_major_formatter(FuncFormatter(format_integer_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}%"))

    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Hit rate by sample size", loc="left", pad=10, weight="bold")
    ax.legend(loc="upper right")

    output_path = FIGURE_DIR / "hit_rates_by_sample_size.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_like_f1_violin(df: pd.DataFrame) -> Path:
    f1_cols = [f"like_F1_{threshold}" for threshold in range(99, 89, -1)]
    data = df.copy()
    data["has_like"] = pd.to_numeric(data["has_like"], errors="coerce")
    data[f1_cols] = data[f1_cols].apply(pd.to_numeric, errors="coerce")
    data = data[data["has_like"] == 1]

    long_df = data[f1_cols].melt(var_name="threshold", value_name="score").dropna()
    threshold_labels = {f"like_F1_{threshold}": f"F1 {threshold / 100:.2f}" for threshold in range(99, 89, -1)}
    long_df["threshold"] = long_df["threshold"].map(threshold_labels)
    ordered_labels = [threshold_labels[f"like_F1_{threshold}"] for threshold in range(99, 89, -1)]

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    sns.violinplot(
        data=long_df,
        x="threshold",
        y="score",
        order=ordered_labels,
        orient="v",
        inner="quartile",
        cut=0,
        linewidth=0.8,
        color="0.72",
        saturation=1,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax.tick_params(axis="both", colors="0.2")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Distribution of F1 scores by threshold", loc="left", pad=10, weight="bold")

    output_path = FIGURE_DIR / "distribution_F1.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def prepare_observations_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["observed_rating"] = pd.to_numeric(data["observed_rating"], errors="coerce")
    data["predicted_rating"] = pd.to_numeric(data["predicted_rating"], errors="coerce")
    data = data.dropna(subset=["observed_rating", "predicted_rating"]).copy()
    data["residual"] = data["predicted_rating"] - data["observed_rating"]
    data["abs_residual"] = data["residual"].abs()
    return data


def _normalized_entropy(series: pd.Series, categories: list[float]) -> float:
    counts = series.value_counts().reindex(categories, fill_value=0).astype(float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts / total
    probs = probs[probs > 0]
    if probs.empty:
        return float("nan")
    entropy = -(probs * probs.map(math.log)).sum()
    return float(entropy / math.log(len(categories)))


def build_user_distribution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    data = prepare_observations_frame(df)
    categories = [0.5 + 0.5 * i for i in range(10)]

    rows = []
    for pseudo, group in data.groupby("pseudo", sort=False):
        ratings = group["observed_rating"].astype(float)
        counts = ratings.value_counts()
        total = len(ratings)
        extreme_frequency = (counts.get(0.5, 0) + counts.get(5.0, 0)) / total if total else float("nan")
        rows.append({
            "pseudo": pseudo,
            "skewness": ratings.skew(),
            "kurtosis": ratings.kurtosis(),
            "extreme_frequency": extreme_frequency,
            "entropy": _normalized_entropy(ratings, categories),
        })

    return pd.DataFrame(rows)


def plot_user_distribution_metrics_vs_r2(obs_df: pd.DataFrame, perf_df: pd.DataFrame) -> Path:
    metrics_df = build_user_distribution_metrics(obs_df)
    data = perf_df[["pseudo", "R2"]].copy()
    data["R2"] = pd.to_numeric(data["R2"], errors="coerce")
    data = data.merge(metrics_df, on="pseudo", how="inner").dropna(subset=["R2", "skewness", "kurtosis", "extreme_frequency", "entropy"])
    data = data[data["R2"] >= 0.10].copy()

    panels = [
        ("skewness", "Skewness", "Skewness"),
        ("kurtosis", "Excess kurtosis", "Kurtosis"),
        ("extreme_frequency", "Extreme frequency", "Extreme frequency"),
        ("entropy", "Normalized entropy", "Entropy"),
    ]

    set_stata_like_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))
    axes = axes.flatten()

    for ax, (column, xlabel, title) in zip(axes, panels):
        panel = data[[column, "R2"]].dropna()
        sns.regplot(
            data=panel,
            x=column,
            y="R2",
            ax=ax,
            scatter_kws={
                "s": 28,
                "color": "0.15",
                "edgecolor": "white",
                "linewidths": 0.3,
                "alpha": 0.9,
            },
            line_kws={"color": "0.35", "linewidth": 1.0},
            ci=None,
            color="0.15",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("R²")
        ax.set_title(title, loc="left", pad=8, weight="bold")
        ax.tick_params(axis="both", colors="0.2")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(format_metric_ticks))

    output_path = FIGURE_DIR / "user_distribution_metrics_vs_r2.svg"
    fig.suptitle("User rating distribution characteristics vs R²", x=0.02, y=0.995, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_parity_observed_predicted(df: pd.DataFrame) -> Path:
    data = prepare_observations_frame(df)
    data["observed_label"] = data["observed_rating"].map(lambda value: f"{value:.1f}")
    ordered_labels = [f"{value:.1f}" for value in sorted(data["observed_rating"].unique())]

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    sns.violinplot(
        data=data,
        x="observed_label",
        y="predicted_rating",
        order=ordered_labels,
        inner="quartile",
        cut=0,
        linewidth=0.8,
        color="0.68",
        saturation=1,
        ax=ax,
    )
    ax.plot(range(len(ordered_labels)), [float(label) for label in ordered_labels], color="0.05", linewidth=1.0, linestyle="--", label="Perfect fit")

    ax.set_ylim(0.4, 5.1)
    ax.set_xlabel("Observed rating")
    ax.set_ylabel("Predicted rating")
    ax.yaxis.set_major_formatter(FuncFormatter(format_metric_ticks))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Parity plot: observed vs predicted ratings", loc="left", pad=10, weight="bold")
    ax.legend(loc="upper left")

    output_path = FIGURE_DIR / "observations_parity_plot.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_residuals_observed(df: pd.DataFrame) -> Path:
    data = prepare_observations_frame(df)
    data["observed_label"] = data["observed_rating"].map(lambda value: f"{value:.1f}")
    ordered_labels = [f"{value:.1f}" for value in sorted(data["observed_rating"].unique())]

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    sns.violinplot(
        data=data,
        x="observed_label",
        y="residual",
        order=ordered_labels,
        inner="quartile",
        cut=0,
        linewidth=0.8,
        color="0.72",
        saturation=1,
        ax=ax,
    )
    ax.axhline(0, color="0.05", linewidth=1.0, linestyle="--", label="Zero residual")

    ax.set_xlabel("Observed rating")
    ax.set_ylabel("Residual (predicted - observed)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}"))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Residuals by observed rating", loc="left", pad=10, weight="bold")
    ax.legend(loc="upper left")

    output_path = FIGURE_DIR / "observations_residuals_by_observed.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_residual_distribution(df: pd.DataFrame) -> Path:
    data = prepare_observations_frame(df)

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    sns.histplot(
        data=data,
        x="residual",
        bins=80,
        stat="density",
        color="0.35",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.2,
        ax=ax,
    )
    ax.axvline(0, color="0.05", linewidth=1.0, linestyle="--", label="Zero")
    ax.axvline(data["residual"].mean(), color="0.35", linewidth=0.9, linestyle=":", label="Mean")

    ax.set_xlabel("Residual (predicted - observed)")
    ax.set_ylabel("Density")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Residual distribution", loc="left", pad=10, weight="bold")
    ax.legend(loc="upper right")

    output_path = FIGURE_DIR / "observations_residual_distribution.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_observed_predicted_histogram(df: pd.DataFrame) -> Path:
    data = prepare_observations_frame(df)

    set_stata_like_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    sns.histplot(
        data=data,
        x="observed_rating",
        bins=10,
        binrange=(0.4, 5.1),
        stat="count",
        color="0.2",
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )

    ax.set_xlim(0.4, 5.1)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(FuncFormatter(format_metric_ticks))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Observed rating distribution", loc="left", pad=10, weight="bold")

    output_path = FIGURE_DIR / "observations_observed_predicted_histogram.svg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    total_users = len(df_perf)
    total_observations = len(df_obs)
    observation_weights = df_perf["observations"]

    user_stats = build_metric_stats(df_perf)
    observation_stats = build_metric_stats(df_perf, weights=observation_weights)
    characteristic_stats = build_characteristic_stats(df_perf)
    characteristic_stats.loc["observations", "total"] = total_observations

    print(f"\nPerformance aggregate weighted by user (N={total_users} users, {total_observations} observations)\n")
    print_metric_block(user_stats, weighted=False)

    print(f"Performance aggregate weighted by observations (N={total_users} users, {total_observations} observations)\n")
    print_metric_block(observation_stats, weighted=True)

    print(f"Sample characteristics (N={total_users} users, {total_observations} observations)\n")
    print_characteristics_block(characteristic_stats)

    figure_path = plot_observations_r2_mean_error(df_perf)
    print(f"\nSaved figure: {figure_path}")

    std_diff_path = plot_std_diff_r2_hit_rate(df_perf)
    print(f"Saved figure: {std_diff_path}")

    hit_rate_path = plot_observations_hit_rates(df_perf)
    print(f"Saved figure: {hit_rate_path}")

    violin_path = plot_like_f1_violin(df_perf)
    print(f"Saved figure: {violin_path}")

    parity_path = plot_parity_observed_predicted(df_obs)
    print(f"Saved figure: {parity_path}")

    residuals_path = plot_residuals_observed(df_obs)
    print(f"Saved figure: {residuals_path}")

    residual_dist_path = plot_residual_distribution(df_obs)
    print(f"Saved figure: {residual_dist_path}")

    histogram_path = plot_observed_predicted_histogram(df_obs)
    print(f"Saved figure: {histogram_path}")

    dist_r2_path = plot_user_distribution_metrics_vs_r2(df_obs, df_perf)
    print(f"Saved figure: {dist_r2_path}")
