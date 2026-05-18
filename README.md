# Regime Switching Models for Time Series

This project demonstrates Markov switching regression models for identifying regime changes in time series data.

## Business context

Some time series do not follow a single pattern. Markets alternate between calm and chaotic. Economies cycle through expansion and recession. Energy demand shifts with seasons, policy changes, and structural shocks. A single model fitted across all of this history will fit none of it well.

Regime switching models solve this by assuming the series operates in distinct states — regimes — each with its own statistical properties. The model learns when the data is in which state and estimates the probability of transitioning between them.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Regime switching functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_samples, regime probabilities, standard deviations)
- Model parameters (number of regimes, switching variance)
- Output settings

## Caveats

- By default, the script generates synthetic data with two regimes.
- The model uses Markov switching regression with switching variance.
- Transition probabilities are estimated from the data.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).