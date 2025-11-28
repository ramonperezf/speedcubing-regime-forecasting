# Speedcubing Regime Modeling and Short-Horizon Ao5 Forecasting

This project models my speedcubing solve times using a Gaussian HMM and a residual neural forecaster. The goal is to understand how my performance evolves across Flow, Baseline, and Tilt states, and to see whether short-horizon changes in my average-of-five (Ao5) can be predicted more accurately than a naive random-walk baseline. Because individual solves are almost pure noise, everything is built around short-window smoothing, state inference, and residual prediction.

---

## Repository Structure
data/
    raw/
    processed/

models/
hmm_gaussian.pkl
hmm_scaler.pkl
hybrid_lstm.pth
lstm_model.py

notebooks/
01_hmm_regime_inference.ipynb
02_regime_diagnostics.ipynb
03_hybrid_lstm_hmm_forecasting.ipynb

results/
transition_matrix.png
dwell_time_distribution.png
pca_regime_projection.png
residual_training_curve.png
residual_forecast_curve.png
final_dynamic_forecast.png
model_diagnostics.txt

README.md


---

## Notebook Summaries

**01 — HMM Regime Inference**  
Cleans the CSTimer export, builds rolling-volatility features, fits a Gaussian HMM, and assigns a regime label (Flow / Baseline / Tilt) to each solve.

**02 — Regime Diagnostics**  
Applies a short smoothing pass to remove single-solve noise, analyzes transition probabilities and dwell patterns, and visualizes how well the regimes separate in PCA space.

**03 — Hybrid LSTM + HMM Forecasting**  
Trains a residual LSTM to predict the next Ao5 *change*, then reconstructs the forecast by adding that residual back to the current Ao5. A dynamic ensemble blends the naive baseline with the LSTM using the HMM’s tilt probability: when instability is high, the model leans more on the LSTM.

---

## Key Results

- **Naive baseline MAE:** ~0.757 s  
- **Residual LSTM MAE:** ~0.727 s  
- **Dynamic ensemble MAE:** ~0.708 s  
- **Lag behavior:** correlation is higher at lag 0 than lag 1, suggesting the final ensemble reacts in real time rather than trailing the sequence.

The improvements are modest, which is expected because Ao5 behaves like a smoothed random walk. The value comes from understanding the regimes, highlighting when performance becomes unstable, and showing that residual modeling only helps when the state is volatile.

---

## How to Run

1. Install dependencies:  
2. Place your CSTimer export in data/raw/ if you would like to run this on your own data
3. Run the notebooks in order (01 → 02 → 03).

All plots and metrics will appear in the results/ folder.

# License

Released under the MIT License.