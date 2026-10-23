import yfinance as yf
import pandas as pd
import numpy as np
import datetime

#aapl = yf.Ticker("AAPL")
#hist1 = aapl.history(period="20y")
#combined = hist1.drop(columns=['Dividends', 'Stock Splits'], axis=1)

combined = pd.DataFrame()

tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOG", "META", "TSLA", "AMD", "CRM"] #, "INTC", "BABA", "PYPL", "TTD", "EA", "ZG", "MTCH", "YELP"]

lenList = []

for stockTicker in tickers:
    stockTicker1 = yf.Ticker(stockTicker)

    hist = stockTicker1.history(period="4y")
    lenList.append(len(hist[:]['Open']))

    final = hist.drop(columns=['Dividends', 'Stock Splits'], axis=1)
    #combined = combined._append(final.iloc[3000:,:])
    combined = combined._append(final)
    print(stockTicker)

adf_i = combined.reset_index()

for i in range(0, len(adf_i["Open"])):
    dateStr = str(adf_i.iloc[i, 0])
    x = datetime.datetime(int(dateStr[0:4]), int(dateStr[5:7]), int(dateStr[8:10]))
    adf_i.iloc[i, 0] = x

from stock_indicators import indicators
from stock_indicators.indicators.common.quote import Quote

#Creating indicators

#Creating quotes list
quotes_list = [
    Quote(d,o,h,l,c,v) 
    for d,o,h,l,c,v 
    in zip(adf_i['Date'], adf_i['Open'], adf_i['High'], adf_i['Low'], adf_i['Close'], adf_i['Volume'])
]


#20 Day SMA
results = indicators.get_sma(quotes_list, lookback_periods=20)
addList = []
for r in results:
    addList.append(r.sma)
adf_i.insert(len(adf_i.columns), "50 day SMA", addList, allow_duplicates=True)

#20 day SMMA
results = indicators.get_smma(quotes_list, lookback_periods=20)
addList = []
for r in results:
    addList.append(r.smma)
adf_i.insert(len(adf_i.columns), "50 day SMMA", addList, allow_duplicates=True)


#RSI
results = indicators.get_rsi(quotes_list, lookback_periods=14)
addList = []
for r in results:
    addList.append(r.rsi)
adf_i.insert(len(adf_i.columns), "RSI", addList, allow_duplicates=True)

#MACD
results = indicators.get_macd(quotes_list)
addList = []
for r in results:
    addList.append(r.macd) #can be changed to signal
adf_i.insert(len(adf_i.columns), "MACD signal", addList, allow_duplicates=True)


#20 day EMA
results = indicators.get_ema(quotes_list, lookback_periods=20)
addList = []
for r in results:
    addList.append(r.ema)
adf_i.insert(len(adf_i.columns), "20 day EMA", addList, allow_duplicates=True)


#OBV - SMA 9 days
results = indicators.get_obv(quotes_list, sma_periods=9)
addList = []
for r in results:
    addList.append(r.obv) #can be changed to obv_sma
adf_i.insert(len(adf_i.columns), "OBV", addList, allow_duplicates=True)

#ADX
results = indicators.get_adx(quotes_list)
addList = []
for r in results:
    addList.append(r.adx)
adf_i.insert(len(adf_i.columns), "ADX", addList, allow_duplicates=True)

#MFI
results = indicators.get_mfi(quotes_list)
addList = []
for r in results:
    addList.append(r.mfi)
adf_i.insert(len(adf_i.columns), "MFI", addList, allow_duplicates=True)

#CMF
results = indicators.get_cmf(quotes_list)
addList = []
for r in results:
    addList.append(r.cmf)
adf_i.insert(len(adf_i.columns), "CMF", addList, allow_duplicates=True)

#SMI
results = indicators.get_smi(quotes_list)
addList = []
for r in results:
    addList.append(r.smi) #can be changed to signal
adf_i.insert(len(adf_i.columns), "SMI", addList, allow_duplicates=True)

print(adf_i)

adf_i.to_csv('stockData.csv')


import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

for i in range(len(tickers)):
    realList = []
    if i % 2 == 0:
        color1 = 'blue'
    else:
        color1 = 'red'
    for f in range(i*lenList[i], i*lenList[i]+lenList[i]):
        realList.append(f)
    #print(adf_i[i*lenList[i]:i*lenList[i]+lenList[i]]['Open'])
    plt.plot(realList, adf_i[i*lenList[i]:i*lenList[i]+lenList[i]]['Open'], label='Actual', color=color1)
plt.suptitle("Real")


plt.show()