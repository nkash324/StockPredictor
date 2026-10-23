import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import datetime


adf_i = pd.read_csv('stockData.csv')

adf_i = adf_i.reset_index(drop = True)

dataStartDay = 50 # min 40
adf = adf_i.iloc[dataStartDay:,:]
adf = adf.drop('Date', axis=1)
adf = adf.reset_index(drop=True)

dayPred = 5
predUseDays = 10
trainLength = 7000
daysOfGraph = 100 #has to be more than predUseDays
testDays = 1000 #has to be more than predUseDays

adf2 = adf[:]
adf2_train = adf2.iloc[0:(trainLength+dayPred),:]
adf2_test = adf2.iloc[trainLength:trainLength+testDays,:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
adf2_train_scaled = sc.fit_transform(adf2_train)
adf2_test_scaled = sc.fit_transform(adf2_test)


sc2 = StandardScaler()
adf2_train_scaled_y = sc2.fit_transform(adf2_train[['Open']])
testThing = sc2.inverse_transform(adf2_train_scaled_y)



xTrain = []
yTrain = []
for i in range(predUseDays,trainLength):
    xTrain.append(adf2_train_scaled[i-predUseDays:i])
    yTrain.append(adf2_train_scaled_y[i:i+dayPred])
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
#print(yTrain)

xTrain_shape = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], xTrain.shape[2]))
#print(xTrain_shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(predUseDays, len(adf.columns))))
model.add(Dropout(0.1))
model.add(LSTM(units=5))
#model.add(Dropout(0.9))
model.add(Dense(dayPred))
model.compile(optimizer='adam',loss='mean_squared_error')




checkpoint_path = "C:/Users/neelk/OneDrive/Documents/Stock Predictor/Checkpoints/cp-{epoch:04d}.ckpt"
model.save_weights(checkpoint_path.format(epoch=0))
checkpoint_dir = os.path.dirname(checkpoint_path)

n_batches = 64
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

model.fit(xTrain_shape,yTrain,epochs=200,batch_size=n_batches, callbacks=[cp_callback])

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################



latest = tf.train.latest_checkpoint(checkpoint_dir)

#model.load_weights(latest)



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

#test = pd.concat([adf2_test, finalOpenPred], axis=1)
realCompare = []
predCompare = []
k = 0
for i in range(0,testDays-predUseDays-dayPred):
    j = i
    for j in range(j,j+dayPred):
        realCompare.append(adf2_test['Open'][j+predUseDays])
    for k in range (0,dayPred):
        predCompare.append(finalOpenPred.iloc[i,k])
#print(realCompare)
#print(predCompare)
d = {'Real': realCompare, 'Pred': predCompare}
fullTest = pd.DataFrame(data=d)

#adf2_test.insert(1, 'Predicted Open', finalOpenPred['Day 1'])
#fullyFinal = pd.concat((finalOpenPred, adf2_test), axis=1)
#print(adf2_test)
#final = adf2_test
adf2_test.to_csv('thing.csv')


graphLim = 100
indexList = []
indexPred = 0
scaleGuessList = []
scaleGuessList = predCompare
#print(predCompare)

plt.rcParams['figure.figsize'] = [24, 20]
figure, axis = plt.subplots(2, 1)
#figure.figure(figsize=(20,20))


realList = []
for i in range(0, graphLim):
    realList.append(i)
axis[0].plot(realList, adf2[0:graphLim]['Open'], label='Actual', color='red')
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

for i in range(0, graphLim-dayPred):
    for k in range(i, i+dayPred):
        indexList.append(k)
    #print(str(i*dayPred) + ":" + str(i*dayPred+dayPred))
    #print(scaleGuessList)
    axis[1].plot(indexList, predCompare[(i*dayPred):(i*dayPred+dayPred)], color='blue')
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

dayOfPredCalc = testDays-predUseDays-dayPred
k = 0
changes = [0]*dayPred
for k in range(0, dayOfPredCalc-1):
    for i in range(0, dayPred-1):
        realChange = fullTest['Real'][k*dayPred+i+1] - fullTest['Real'][k*dayPred] # +i]
        predChange = fullTest['Pred'][k*dayPred+i+1] - fullTest['Pred'][k*dayPred] # +i]
        #print(str(i) + ":" + str(realChange) + "," + str(predChange))
        if (realChange <= 0 and predChange <= 0) or (realChange >= 0 and predChange >= 0):
            changes[i] += 1
for i in range(0, dayPred-1):
    print("Day " + str(i + 1) + " Accuracy: " + str(changes[i]/dayOfPredCalc))

plt.legend()

plt.show()

#next step is to take absolute value of change and see if each pred ends higher or lower and also if the real ends higher or lower then calculate percentage