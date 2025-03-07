import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
X = data[['petal length (cm)']]
y = data['sepal length (cm)']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X)
print(y_pred)


plt.figure(figsize=(10,6))
plt.scatter(X,y,color='blue',label = 'Veri Noktaları')
plt.plot(X,y_pred, color = 'red',linewidth=2,label='Regresyon Eğrisi')
plt.xlabel('Petal Lenghth (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Petal Length vs Sepal Length : Regresyon Eğrisi')
plt.legend()
plt.show()
plt.show()
