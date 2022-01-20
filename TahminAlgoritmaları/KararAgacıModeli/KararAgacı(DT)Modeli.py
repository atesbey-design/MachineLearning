
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
#Verinin dahil edilmesi Slice edilmesi
veriler=pd.read_csv("maaslar_yeni.csv")
print(veriler)
print(veriler.corr())

x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]

X=x.values
Y=y.values


#Karar ağacı yapısı oluşturma
from sklearn.tree import DecisionTreeRegressor

decisionTree=DecisionTreeRegressor(random_state=0)
decisionTree.fit(X,Y)

print("Decision Tree OLS ")
model=sm.OLS(decisionTree.predict(X),X)


print(model.fit().summary())


print("Decision Tree R2 Değeri")
print((r2_score(Y,decisionTree.predict(X))))