import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import statsmodels.api as sm
veriler =pd.read_csv("maaslar_yeni.csv")

#Gerekli verileri slice ettik
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values
#Bize gerekli olan bilgileri analiz etmek için arasındaki kolerasyona baktık
#Şayet iki veri arasında kolerasyon yüksekse bu iki veriden birisi Dummy Variable yerine geçmektedir

print(veriler.corr())

#linear Regression Modeli Oluşturmak
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Model hakkında kapsamlı bilgi edinmek
model=sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
#Modelin Skorunu Analiz etmek
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))
