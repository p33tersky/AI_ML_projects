# import numpy as np
# import matplotlib.pyplot as plt
# X1 = np.array([[1,2], [1,3], [2,3], [2,4], [3,1]])
# X2 = np.array([[10,15], [11,16], [12,17], [11,13], [12,13]])

# plt.scatter(X1[:,0], X1[:,1]) # w pythonie domyślnie próbki zapisuje się w wierszach macierzy
# plt.scatter(X2[:,0], X2[:,1]) # więc X[:,0] to wsp. 'x-owe' wszystkich próbek
# X = np.vstack((X1,X2))

# # Dodanie do każdej próbki dodatkowego elementu '1': model powierzchni to  Ax + By + C = 0
# X = np.c_[X, np.ones(X.shape[0])]
# print(X.shape)
# t = np.ones(X.shape[0])
# print(t)
# # próbki klasy 1 będą miały etykiety równe '-1'
# t[:X1.shape[0]] *= -1
# Theta = np.linalg.inv(X.T@X)@X.T@t
# print(Theta)
# # narysowanie powierzchni decyzyjnej: określenie dziedziny (100 punktów o odciętych pokrywających cały zakres)
# x = np.linspace(np.min(X1[:,0]), np.max(X2[:,1]),100)
# # Równanie to Ax+By+C=0, parametry to [theta0 = A, theta_1=B, theta_2=C], więc predykcja to y = 1/B(-Ax-C) 
# plt.figure()
# plt.scatter(X1[:,0], X1[:,1]) # w pythonie domyślnie próbki zapisuje się w wierszach macierzy
# plt.scatter(X2[:,0], X2[:,1]) # więc X[:,0] to wsp. 'x-owe' wszystkich próbek
# plt.plot(x,(-Theta[0]*x - Theta[2])/Theta[1], color='green') # rysowana jest linia o równaniu y = -x

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

# Wygenerowanie danych
# Załóżmy, że nasze dane mają dwie cechy
# Pierwsza klasa
X1 = np.random.normal(loc=[1, 1], scale=[1, 1], size=(100, 2))
print(X1)
y1 = np.zeros(100)

# Druga klasa
X2 = np.random.normal(loc=[3, 3], scale=[1, 1], size=(100, 2))
y2 = np.ones(100)

# Połączenie danych
X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wytrenowanie modelu
ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_scaled, y)

# Obliczenie współczynników prostej separującej
coef = ridge_classifier.coef_[0]
intercept = ridge_classifier.intercept_

# Wykres danych
plt.figure(figsize=(8, 6))

# Punkty pierwszej klasy
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='blue', label='Klasa 0')
# Punkty drugiej klasy
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='red', label='Klasa 1')

# Prosta separująca
x_vals = np.linspace(-3, 3, 100)
y_vals = -(coef[0] * x_vals + intercept) / coef[1]
plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Prosta separująca')

plt.xlabel('Cecha 1 (standaryzowana)')
plt.ylabel('Cecha 2 (standaryzowana)')
plt.title('Klasyfikacja liniowa z użyciem Ridge Classifier')
plt.legend()
plt.grid(True)
plt.show()
