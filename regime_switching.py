import logging
from pathlib import Path

import pandas as pd
from src.core import (
    add_predictions,
    calculate_accuracy,
    calculate_regime_durations,
    calculate_regime_statistics,
    fit_markov_switching,
    generate_regime_data,
    plot_confusion_matrix,
    plot_density_distribution,
    plot_regime_comparison,
    plot_regime_data,
    plot_transition_matrix,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_model_fit(result) -> None:
    logging.info(f"\n{result.summary()}")
    logging.info("\nTransition Matrix:")
    logging.info(f"\n{result.regime_transition}")


def log_evaluation_metrics(df: pd.DataFrame) -> None:
    logging.info("\nModel Performance Metrics:")
    logging.info(f"Prediction Accuracy: {calculate_accuracy(df):.2%}")
    logging.info("\nRegime Statistics:")
    for regime, regime_stats in calculate_regime_statistics(df).items():
        logging.info(f"\nRegime {regime}:")
        logging.info(f"Mean: {regime_stats['mean']:.2f}")
        logging.info(f"Std: {regime_stats['std']:.2f}")
        logging.info(f"Skewness: {regime_stats['skewness']:.2f}")
        logging.info(f"Kurtosis: {regime_stats['kurtosis']:.2f}")

    logging.info("\nAverage Duration in Each Regime:")
    for regime, duration in calculate_regime_durations(df).items():
        logging.info(f"Regime {regime}: {duration:.2f} periods")

    transitions = pd.DataFrame(
        {"From": df["Predicted_Regime"][:-1], "To": df["Predicted_Regime"][1:]}
    )
    logging.info("\nTransition Counts:")
    logging.info(f"\n{pd.crosstab(transitions['From'], transitions['To'])}")


def save_analysis_plots(df: pd.DataFrame, result, output_dir: Path = Path(".")) -> None:
    plot_regime_data(df, output_dir / "original_data_regimes.png", plot=True)
    plot_regime_comparison(df, output_dir / "true_vs_predicted_regimes.png", plot=True)
    plot_density_distribution(df, output_dir / "density_distribution.png", plot=True)
    plot_transition_matrix(result, output_dir / "transition_matrix.png", plot=True)
    plot_confusion_matrix(df, output_dir / "confusion_matrix.png", plot=True)


def main() -> None:
    configure_logging()
    df = generate_regime_data()
    result = fit_markov_switching(df["Data"].values)
    log_model_fit(result)
    df = add_predictions(df, result)
    save_analysis_plots(df, result)
    log_evaluation_metrics(df)


if __name__ == "__main__":
    main()
