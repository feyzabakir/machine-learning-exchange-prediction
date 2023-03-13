
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_csv("2016dolaralis.csv")

x = veri["Gun"]
y = veri["Fiyat"]

x = np.array(x).reshape(251, 1)  #boyutlandırma
y = np.array(y).reshape(251, 1)

plt.scatter(x, y)   #nokta nokta verileri çizdirir
plt.show()

# Lineer Reg.
tahminlineer = LinearRegression()
tahminlineer.fit(x, y)    # verileri x ve ye eksenine ekliyorum.
tahminlineer.predict(x)   #günlere göre fiyat tahmini yapıyoruz

plt.scatter(x, y)
plt.plot(x, tahminlineer.predict(x), c="red")
plt.show()

# Polinom Reg.
tahminpolinom = PolynomialFeatures(degree=3)
Xyeni = tahminpolinom.fit_transform(x)

polinommodel = LinearRegression()
polinommodel.fit(Xyeni, y)
polinommodel.predict(Xyeni)

plt.scatter(x, y)
plt.plot(x, tahminlineer.predict(x), c="red")
plt.plot(x, polinommodel.predict(Xyeni), c="green")
plt.show()


hatakaresilineer = 0
hatakaresipolinom = 0

for i in range(len(Xyeni)):
    hatakaresipolinom = hatakaresipolinom + (float(y[i]) - float(polinommodel.predict(Xyeni)[i])) ** 2

for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (float(y[i]) - float(tahminlineer.predict(x)[i])) ** 2



hata=99999
index=0

plt.scatter(x, y)
plt.plot(x, tahminlineer.predict(x), c="red")
plt.plot(x, polinommodel.predict(Xyeni), c="green")

for a in range(120):
    hatakaresipolinom = 0
    tahminpolinom = PolynomialFeatures(degree=a+1)
    Xyeni = tahminpolinom.fit_transform(x)

    polinommodel = LinearRegression()
    polinommodel.fit(Xyeni,y)
    polinommodel.predict(Xyeni)

    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i])-float(polinommodel.predict(Xyeni)[i]))**2

    print(a + 1, "inci dereceden fonksiyonda hata,", hatakaresipolinom)

    if hatakaresipolinom < hata:
        hata = hatakaresipolinom
        index=a+1
print("")
print("hesaplanan en düşük hata", hata)
print("hesaplanan en düşük hata derecesi", index)

tahminpolinom8 = PolynomialFeatures(degree=index)
Xyeni = tahminpolinom8.fit_transform(x)

polinommodel8 = LinearRegression()
polinommodel8.fit(Xyeni, y)
polinommodel8.predict(Xyeni)


plt.plot(x, polinommodel8.predict(Xyeni),c="orange")

plt.show()
for i in range(250):
    asilDeger=float(y[i])
    tahminDeger=float(polinommodel8.predict(Xyeni)[i])

    print(f"{i}.gündeki dolar değeri",asilDeger )
    print(f"{i}.gündeki tahmin edilen dolar değeri", tahminDeger)
    print("Hesaplana hata", asilDeger - tahminDeger)  # dolar tah. hata
    print("")

