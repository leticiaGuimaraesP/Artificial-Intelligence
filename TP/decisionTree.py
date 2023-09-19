import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree

#Leitura do Arquivo CSV:
base = pd.read_csv('heart.csv')

#Separação de Atributos de Entrada e Classe:
X_prev = base.iloc[:, 0:11].values 
y_classe = base.iloc[:, 11].values

#Tratamento de dados categóricos
lb = LabelEncoder() 
X_prev[:,0] = lb.fit_transform(X_prev[:,0]) #Age
X_prev[:,1] = lb.fit_transform(X_prev[:,1]) #Sex
X_prev[:,3] = lb.fit_transform(X_prev[:,3]) #RestingBP
X_prev[:,4] = lb.fit_transform(X_prev[:,4]) #Cholesterol
X_prev[:,5] = lb.fit_transform(X_prev[:,5]) #FastingBS
X_prev[:,7] = lb.fit_transform(X_prev[:,7]) #MaxHR
X_prev[:,9] = lb.fit_transform(X_prev[:,9]) #Oldpeak

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [2, 6, 8, 10])], remainder='passthrough') #Determina a coluna que o OneHotEncoder será aplicado
X_prev = onehotencoder_restaurante.fit_transform(X_prev)


X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

#Teste do Modelo
prevision = modelo.predict(X_teste)

#Gera a matrix de confusão
confusion_matrix(y_teste, prevision)
cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, prevision))
print(accuracy_score(y_teste, prevision))

#Plotando a árvore
tree.plot_tree(Y)
plt.show()