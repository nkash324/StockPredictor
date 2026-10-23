import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import datetime


adf_i = pd.read_csv('stockData.csv')

for i in range(0, len(adf_i["Open"])):
    dateStr = adf_i.iloc[i, 0]
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
    addList.append(r.signal)
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
    addList.append(r.obv_sma)
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
    addList.append(r.signal)
adf_i.insert(len(adf_i.columns), "SMI", addList, allow_duplicates=True)



dataStartDay = 40 # min 40
adf = adf_i.iloc[500:,:]
adf = adf.drop('Date', axis=1)
adf = adf.reset_index(drop=True)

dayPred = 7
predUseDays = 10
trainLength = 3500
daysOfGraph = 50 #has to be more than predUseDays
testDays = 500 #has to be more than predUseDays

adf2 = adf.iloc[:]
adf2_train = adf2.iloc[0:(trainLength+dayPred),:]
adf2_test = adf2.iloc[trainLength:trainLength+testDays,:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
adf2_train_scaled = sc.fit_transform(adf2_train)
adf2_test_scaled = sc.fit_transform(adf2_test)
#print(adf2_train_scaled)

sc2 = StandardScaler()
adf2_train_scaled_y = sc2.fit_transform(adf2_train[['Open']])

#print(adf2_train_scaled_y)

xTrain = []
yTrain = []
for i in range(predUseDays,trainLength):
    xTrain.append(adf2_train_scaled[i-predUseDays:i])
    yTrain.append([adf2_train_scaled_y[i], adf2_train_scaled_y[i+dayPred]])
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
#print(len(yTrain))
#print(len(adf2_train_scaled_y))


xTrain_shape = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], xTrain.shape[2]))

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(predUseDays, len(adf2.columns))))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
#model.add(Dropout(0.9))
model.add(Dense(2))
model.compile(optimizer='adam',loss='mean_squared_error')




checkpoint_path = "C:/Users/Neel Kashyap/Documents/Test Code/Checkpoints/cp-{epoch:04d}.ckpt"
model.save_weights(checkpoint_path.format(epoch=0))
checkpoint_dir = os.path.dirname(checkpoint_path)

n_batches = 32
stepsPerBatch = math.ceil(trainLength//n_batches)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=0, 
    save_weights_only=True,
    save_freq=7)

#find the correct save_freq length, calculate probability of luck, add sentiment analysis from website, tested all epchs sizes excpet inputs

model.save_weights(checkpoint_path.format(epoch=0))



###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

#model.fit(xTrain_shape,yTrain,epochs=20,batch_size=n_batches, callbacks=[cp_callback])

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################



latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)



adf_train_lastNum = adf2_train.iloc[-predUseDays:]
adf_train_lastNum.to_csv('test_start_trans_init.csv')
adf_test_full = adf2_test
full_adf = pd.concat((adf_train_lastNum,adf_test_full), axis=0)

full_adf = sc.transform(full_adf)

xTrainShapePred = []
for i in range(predUseDays,testDays):
    xTrainShapePred.append(full_adf[i-predUseDays:i])
xTrainShapePred = np.array(xTrainShapePred)


yTest = model.predict(xTrainShapePred)
yFinalPred = sc2.inverse_transform(yTest)
#for i in range(0,20):
#    print(str(yTest[i][0]) + ':' + str(yFinalPred[i][0]))


finalOpenPred = pd.DataFrame(yFinalPred)
#print(finalOpenPred)




adf2_test = adf2_test.reset_index()
finalOpenPred.to_csv('predEndData.csv')
adf2_test.to_csv('test.csv')

#print(finalOpenPred)
#test = pd.concat([adf2_test, finalOpenPred], axis=1)
realCompare = []
predCompare = []
for i in range(0, len(finalOpenPred[0])):
    predCompare.append(finalOpenPred.iloc[i][0])
    predCompare.append(finalOpenPred.iloc[i][1])
for i in range(0,testDays-predUseDays):
    realCompare.append(adf2_test['Open'][i])
    realCompare.append(adf2_test['Open'][i+dayPred])

#print(realCompare)
#print(predCompare)
#print(len(realCompare))
#print(len(predCompare))
d = {'Real': realCompare, 'Pred': predCompare}
fullTest = pd.DataFrame(data=d)
#print(d)

#adf2_test.insert(1, 'Predicted Open', finalOpenPred['Day 1'])
#fullyFinal = pd.concat((finalOpenPred, adf2_test), axis=1)
#print(adf2_test)
#final = adf2_test
fullTest.to_csv('thing.csv')


graphLim = 100
indexList = []
indexPred = 0
scaleGuessList = []
scaleGuessList = predCompare
#print(predCompare)

figure, axis = plt.subplots(2, 1)

realList = []
for i in range(0, graphLim):
    realList.append(i)
axis[0].plot(realList, adf2_test[0:graphLim]['Open'], label='Actual', color='red')
axis[0].set_title("Real")

#scaling

'''
k = 0
scaleGuess = 80
scaleGuessList = []
for i in range(0,len(predCompare)):
    scaleGuessList.append(predCompare[i] + scaleGuess)
'''

#Messy, every day new pred

dayOfPredCalc = testDays-predUseDays
shift = realCompare[0] - predCompare[0]
for i in range(len(predCompare)):
    predCompare[i] += shift
addList = []
for i in range(0, graphLim):
    indexList.append(i)
    indexList.append(i+dayPred)
    #print(str(i*dayPred) + ":" + str(i*dayPred+dayPred))
    #print(indexList)
    addList = predCompare[i*2:i*2+2]
    predChange = fullTest['Pred'][i*2+1] - fullTest['Pred'][i*2]
    if (predChange > 0):
        axis[1].plot(indexList, addList, color='green')
    else:
        axis[1].plot(indexList, addList, color='red')
    indexList = []
axis[1].set_title("Prediction")

#More clean, but only preds for next # Pred days every interval of Pred days
'''
for i in range(0, math.trunc(graphLim/dayPred)):
    for k in range(indexPred, indexPred+dayPred):
        indexList.append(k)
    plt.plot(indexList, scaleGuessList[(indexPred*dayPred):(indexPred*dayPred+dayPred)], color='blue')
    indexPred += dayPred
    indexList = []
'''
    
#Every one day after prediction data
'''
for i in range(0, graphLim):
    indexList.append(scaleGuessList[indexPred])
    indexPred += dayPred
print(indexList)
plt.plot(indexList, label = 'predicted', color='blue')
'''

fullTest.to_csv('testEndData.csv')

k = 0
changes = 0
for i in range(0, dayOfPredCalc-1, 2):
    realChange = fullTest['Real'][i+1] - fullTest['Real'][i]
    predChange = fullTest['Pred'][i+1] - fullTest['Pred'][i]
    #print(str(i) + ":" + str(realChange) + "," + str(predChange))
    if (realChange <= 0 and predChange <= 0) or (realChange >= 0 and predChange >= 0):
        changes += 1
print("Day #" + str(dayPred) + " Accuracy: " + str(changes/dayOfPredCalc*2))

plt.legend()

plt.show()

#next step is to take absolute value of change and see if each pred ends higher or lower and also if the real ends higher or lower then calculate percentage