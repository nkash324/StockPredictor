# Stock Predictor (2023–24)

LSTM model forecasting stock prices, from my independent research paper *"Comparing the effectiveness of using qualitative and quantitative data in combination to predict stock movement against standalone methods."*

## Approach
- **Price model:** an LSTM (Keras/TensorFlow) trained on historical daily prices plus technical indicators (SMA, SMMA, RSI, MACD, EMA, OBV, ADX, MFI, CMF, SMI), using a 10-day input window.
- **Sentiment:** news and social-media sentiment scores combined with the price data.
- **Results (from the paper):** ~58% directional accuracy over 3–5 week horizons for the combined model, vs. ~52% from price data alone at longer horizons.

## Files
- `stock_creatorr.py` downloads price history with yfinance and computes the indicators into `stockData.csv`
- `stockInfoTest.py` / `oneDayPred.py` train the LSTM, then predict on a held-out period and score direction

## Run
```
pip install yfinance pandas numpy scikit-learn tensorflow stock-indicators matplotlib
python stock_creatorr.py
python oneDayPred.py
```
