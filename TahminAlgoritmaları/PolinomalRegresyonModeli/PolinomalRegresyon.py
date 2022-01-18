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



#Polinomal Regresyon modeli inşaa edilmesi
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression
d=2
PolynomialRegression=PolynomialFeatures(degree=d)
X_poly=PolynomialRegression.fit_transform(x)

#Eğitilen yapının lineer modele indirgenmesi
linearReg=LinearRegression()
linearReg.fit(X_poly,Y)


#Tahmin aşaması
PolRegModel=sm.OLS(linearReg.predict(PolynomialRegression.fit_transform(x)),x)
print(PolRegModel.fit().summary())

print("Polinomal Regresyon R2 değeri")
print(r2_score(y,linearReg.predict(PolynomialRegression.fit_transform(x))))


