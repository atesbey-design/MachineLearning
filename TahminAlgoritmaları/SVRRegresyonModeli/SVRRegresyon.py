import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import statsmodels.api as sm
veriler =pd.read_csv("maaslar_yeni.csv")
X=veriler.iloc[:,2:5]
x=X.values
Y=veriler.iloc[:,5:]
y=Y.values
#Veri önişleme aşamaları
print(veriler.iloc[:,2:5])
print(veriler.iloc[:,5:].values)
print(veriler.corr())
print(veriler.isna().sum())

#Support Vector Regression modeli için verilerin 0 ile 1 aralığına indirgenmesi
from sklearn.preprocessing import StandardScaler

scaler1=StandardScaler()
x_olcek=scaler1.fit_transform(x)
scaler2=StandardScaler()
y_olcek=np.ravel(scaler2.fit_transform((y.reshape(-1,1))))
#SVR modelini inşa etmek
from sklearn.svm import SVR

svrRegression=SVR(kernel="rbf")
svrRegression.fit(x_olcek,y_olcek)



#SVR modelinin performansını ölçmek
print(("SVR DURUM"))
model=sm.OLS(svrRegression.predict(x_olcek),x_olcek)
print(model.fit().summary())


print("SVR R2 değeri")
print((r2_score(y_olcek,svrRegression.predict(x_olcek))))