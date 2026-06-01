# Regime Switching Models for Time Series Analysis in Python

Some time series do not follow a single pattern. Markets alternate between calm and chaotic. Economies cycle through expansion and recession. Energy demand shifts with seasons, policy changes, and structural shocks. A single model fitted across all of this history will fit none of it well.

Regime switching models solve this by assuming the series operates in distinct states — regimes — each with its own statistical properties. The model learns when the data is in which state and estimates the probability of transitioning between them.

## What Are Regime Switching Models?

Regime switching models, introduced by James Hamilton in 1989, capture structural changes in time series by allowing transitions between different states. They are particularly useful in:

1. **Economics** — modeling recessions vs. expansions
2. **Finance** — identifying bull and bear markets
3. **Energy** — detecting shifts in demand or production trends
4. **Weather** — capturing transitions between climatic states

The most common implementation is the **Markov Switching Model**, where transitions between regimes follow a Markov process: the probability of being in a regime tomorrow depends only on the regime today, not on the full history.

## How Markov Switching Works

Think of the economy switching between boom and bust. At any moment, it follows patterns specific to its current state. During a boom, growth might average 3% with low volatility. During a bust, growth might average -1% with high volatility.

The Markov assumption means the transition probabilities are fixed:
- P(boom → boom) = 0.90, P(boom → bust) = 0.10
- P(bust → bust) = 0.70, P(bust → boom) = 0.30

The model estimates these transition probabilities from data alongside the state-specific means and variances. You do not need to label which periods were which regime — the model figures that out.

## Implementation in Python

The `statsmodels` library provides `MarkovRegression` for this. Here we generate synthetic data with two regimes (low and high volatility) and fit a two-state switching model:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

np.random.seed(42)

n = 500
regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
data = np.array([
    np.random.normal(0, 1) if r == 0 else np.random.normal(0, 5)
    for r in regimes
])

model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
print(result.summary())
```

Setting `switching_variance=True` allows each regime to have its own variance — essential for volatility modeling. Without it, the model only allows regime-specific means.

## Reading the Results

The fitted model gives you several outputs:

**Transition matrix** — the estimated probabilities of moving between regimes:

```python
print(result.regime_transition)
```

A typical output for this synthetic data might show:
- Regime 0 (low volatility): 88% chance of staying in regime 0
- Regime 1 (high volatility): 62% chance of staying in regime 1

High volatility regimes tend to be more transient — they spike and resolve. Low volatility regimes persist.

**Smoothed marginal probabilities** — for each time point, the posterior probability of being in each regime:

```python
probs = result.smoothed_marginal_probabilities
predicted_regime = np.argmax(probs, axis=1)
```

This lets you label each observation with its most likely regime and compare against the true labels. On synthetic data with known ground truth, you can construct a confusion matrix:

```python
import pandas as pd
confusion = pd.crosstab(regimes, predicted_regime)
print(confusion)
```

In practice, the model recovers the high-volatility regime well when the variance difference between states is large. As the states become more similar, accuracy drops — which is expected and a good reason to validate the number of regimes before committing.

## Choosing the Number of Regimes

Two regimes is the natural starting point. To decide if you need more, compare AIC and BIC across models:

```python
for k in [2, 3, 4]:
    m = MarkovRegression(data, k_regimes=k, trend='c', switching_variance=True)
    r = m.fit(disp=False)
    print(f"k={k}  AIC={r.aic:.1f}  BIC={r.bic:.1f}")
```

Lower is better. In practice, three or four regimes rarely adds meaningful interpretability over two, especially when the additional states do not correspond to anything in the domain.

## Practical Considerations

**Stationarity** — like other time series models, regime switching assumes the data is stationary within each regime. Test with ADF or KPSS before fitting and difference if needed.

**Initialization sensitivity** — the EM algorithm used to fit these models can get stuck in local optima. Fit multiple times with different starting values and take the best log-likelihood.

**Interpretability** — regime labels are arbitrary. Regime 0 might be high volatility in one run and low volatility in another. Always check the regime-specific means and variances, not just the label numbers.

## Conclusion

Regime switching models are one of the cleaner ways to handle structural change in time series without manually segmenting the data. Instead of choosing a break point by eye, the model estimates transition probabilities from data and gives you posterior probabilities for each regime at each point in time.

The main use cases:
- **Forecasting** — fit regime-specific models and weight predictions by the current regime probability
- **Change point detection** — identify when the series shifted character
- **Feature engineering** — add regime labels as features for downstream classification or clustering

The `statsmodels` implementation is straightforward for two-regime models. For more complex specifications — time-varying transition probabilities, regime-dependent AR terms — you will need to build on it or move to a more specialized library.
