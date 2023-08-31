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
base = pd.read_csv('restaurante.csv', usecols=['Instancia', 'Alternativo', 'Bar','Sex/Sab','Fome','Cliente','Preco','Chuva','Res','Tipo','Tempo','Conclusao'])

#Separação de Atributos de Entrada e Classe:
X_prev = base.iloc[:, 0:11].values #Seleciona todas as linhas do DataFrame e extrai as colunas de 0 até 9
y_classe = base.iloc[:, 11].values

#Tratamento de dados categóricos
lb = LabelEncoder() #usado para transformar rótulos de classes ou categorias em números inteiros

X_prev[:,0] = lb.fit_transform(X_prev[:,1]) 
X_prev[:,1] = lb.fit_transform(X_prev[:,2]) 
X_prev[:,2] = lb.fit_transform(X_prev[:,3]) 
X_prev[:,3] = lb.fit_transform(X_prev[:,4])
X_prev[:,4] = X_prev[:,5]
X_prev[:,5] = X_prev[:,6]
X_prev[:,6] = lb.fit_transform(X_prev[:,7]) 
X_prev[:,7] = lb.fit_transform(X_prev[:,8]) 
X_prev[:,8] = X_prev[:,9]
X_prev[:,9] = X_prev[:,10]
X_prev = X_prev[:, :-1] #Apaga a ultima coluna (10), que agora esta alocada na posição 9
print(X_prev)
#Binarizar atributos não ordinais - OneHotEncoder

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [4, 5, 8, 9])], remainder='passthrough') #Determina a coluna que o OneHotEncoder será aplicado
X_prev = onehotencoder_restaurante.fit_transform(X_prev) #Aplica o ColumnTransformer aos dados de entrada

print(X_prev)

X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

#Teste do Modelo
previsoes = modelo.predict(X_teste)
print("Previsao:", previsoes) #['Sim' 'Sim' 'Nao']
print("Validacao: ", y_teste) #['Sim' 'Nao' 'Sim']
print(classification_report(y_teste, previsoes))

#print(y_teste)
#print(accuracy_score(y_teste,previsoes))

#Gera a matrix de confusão
confusion_matrix(y_teste, previsoes)
cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

#Plotando a árvore
tree.plot_tree(Y)
plt.show()