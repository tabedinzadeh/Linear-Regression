
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 
import csv

df = pd.read_csv('imports.csv')

print(df.describe())
cdf=df[['width','weight']]
print(cdf)

plt.scatter(df.width ,df.weight , color='blue')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()

msk = np.random.rand(len(df)) <0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter( train.width, train.weight , color='blue')
plt.xlabel('width')
plt.ylabel('weight')
plt.show() 

from sklearn import linear_model
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['width']])
train_y = np.asanyarray(train[['weight']])
reg.fit ( train_x , train_y)

print('coefs:', reg.coef_)
print('inter:' , reg.intercept_)

plt.scatter(train.width ,train.weight , color='blue')
plt.plot( train_x , reg.coef_[0][0]*train_x + reg.intercept_[0] , '-r')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['width']])
test_y = np.asanyarray(test[['weight']])
test_y_= reg.predict(test_x)

print(" mean ab error: %.2f" % np.mean( np.absolute(test_y_ - test_y))) 
print(" residule sum of sq: %.2f(MSE)" % np.mean((test_y_ - test_y)**2)) 
print(" R2-score: %.2f" % r2_score(test_y , test_y_))




