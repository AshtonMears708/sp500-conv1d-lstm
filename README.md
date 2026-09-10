# Conv1D-LSTM Stock Price Forecasting

A hybrid convolutional-recurrent network forecasting S&P 500 and Amazon (AMZN) closing prices
from OHLCV data. **Test MAPE: 1.646%** on the S&P 500.

Built at Western Carolina University, spring 2026.

---

## Why a hybrid

A plain LSTM has to learn both short-range shape and long-range dependency in the same
recurrent weights. Splitting the job works better:

- A **causal Conv1D** layer extracts local price patterns across the window. Causal padding
  matters — it prevents the convolution from seeing future timesteps, which would leak the
  target.
- **Stacked LSTMs** then carry sequential context across the window.

```
Conv1D(64, kernel=5, causal, relu)
LSTM(128, return_sequences, L2 regularized)
Dropout(0.2)
LSTM(64)
Dropout(0.2)
Dense(32, relu)
Dense(1)
```

151,681 trainable parameters.

---

## Training

- **Learning rate chosen, not guessed.** A log-scale range sweep from 1e-6 to 1e-1 across 100
  epochs identifies where loss falls fastest before diverging; the run then uses 7e-5.
- Adam, MSE loss, up to 200 epochs.
- `EarlyStopping` on validation loss, patience 20, with `restore_best_weights=True` so the
  saved model is the best epoch rather than the last.
- Dropout at 0.2 after each LSTM plus L2 on the first, against a small dataset and a model
  that will happily memorize it.
- Trained on 2× NVIDIA T4 GPUs (Kaggle).

## Results

| Metric | S&P 500 test |
|---|---|
| MAPE | **1.646%** |
| MAE  | 97.699 |
| RMSE | 122.042 |

Predictions are inverse-transformed from MinMax-scaled space back to price space before
metrics are computed, so these are in dollars, not scaled units.

## Honest limitations

- Single train/test split on one historical period. No walk-forward or rolling-origin
  validation, so these numbers should be read as one result, not a robust performance estimate.
- Low MAPE on daily closes is easier than it sounds — prices are highly autocorrelated and a
  naive "tomorrow equals today" predictor already scores well. A persistence baseline is the
  honest comparison and is not yet included here.
- **Not investment advice, and not a trading system.** This is a sequence-modeling exercise.

## Layout

```
notebooks/sp500-conv1d-lstm.ipynb    Main model and results
notebooks/amzn-conv1d-lstm.ipynb     Same architecture on AMZN
notebooks/earlier-config-window120.ipynb  Earlier run, 120-day window
```

## Credits

Team project with **Wyatt** and **Kai**.
