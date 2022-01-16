#Veriseti bilkav.com'dan alınmıştır.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Verileri dahil etme
data=pd.read_csv("maaslar.csv")
print(data)

egitimSev=data.iloc[:,1:2]
maas=data.iloc[:,2:3]
EgitimSev=egitimSev.values
Maas=maas.values

from sklearn.preprocessing import StandardScaler
#Ölçeklendirme
sc1=StandardScaler()
egitimSev_sc=sc1.fit_transform(EgitimSev)
sc2=StandardScaler()
maas_sc=np.ravel(sc2.fit_transform(Maas.reshape(-1,1)))
print(maas_sc)

from sklearn.svm import SVR
#Support Vector Regresion ile çalışmak
supportVector=SVR(kernel="rbf")
supportVector.fit(egitimSev_sc,maas_sc)


#Grafik arayüzü
plt.scatter(egitimSev_sc,maas_sc,color="red")
plt.plot(egitimSev_sc,supportVector.predict(egitimSev_sc),color="blue")
plt.show()
