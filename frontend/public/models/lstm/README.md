# Attention-LSTM Model Files

This directory holds the TensorFlow.js model artifacts for browser-side Attention-LSTM inference.

## Why these files are here

The Attention-LSTM runs **in the user's browser** (not on the server) to stay within Railway free tier 512MB RAM.
The server prepares features and returns them; the browser loads these model files and runs MC Dropout inference.

## Generating the model

Run once locally (requires TensorFlow + tensorflowjs):

```bash
cd backend
pip install tensorflow tensorflowjs scikit-learn pandas yfinance
python scripts/train_lstm.py
```

This will:
1. Download 10 years of price data for a diverse 15-asset universe
2. Train the Attention-LSTM(64) + Bahdanau attention on chronological 70/15/15 split
3. Save `backend/scripts/lstm_model.h5`
4. Export to TF.js format → `frontend/public/models/lstm/` (this directory)

## Committing the model

After training, commit the generated files:
```bash
git add frontend/public/models/lstm/
git commit -m "Add pre-trained Attention-LSTM TF.js model"
```

Vercel serves them as static assets. Model size: ~300-600 KB.

## Architecture

- Input: (60 timesteps, 5 features) — [return, vol_21d, mom_5d, mom_21d, rsi_14], normalized to [-1, 1]
- LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.10)
- Bahdanau additive attention → context vector ∈ ℝ⁶⁴
- Dropout(0.20) → Dense(32, relu) → Dense(1)
- MC Dropout: 200 stochastic forward passes at inference (training=True)

## References

- Bahdanau, Cho & Bengio (2015, ICLR) — attention mechanism
- Gal & Ghahramani (2016, ICML) — MC Dropout for Bayesian uncertainty
- CS230 Stanford (2020) — temporal attention for stock prediction
