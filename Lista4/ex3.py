import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree

# Criando o dataframe de exemplo
df = pd.read_csv('Lista4/jogar.csv')

X_prev = df.iloc[:, 0:4].values #Seleciona todas as linhas do DataFrame e extrai as colunas de 0 até 9
y_classe = df.iloc[:, 4].values 

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3])], remainder='passthrough') #Determina a coluna que o OneHotEncoder será aplicado
X_prev = onehotencoder_restaurante.fit_transform(X_prev) #Aplica o ColumnTransformer aos dados de entrada

X_train, X_test, y_train, y_test = train_test_split(X_prev, y_classe, test_size=0.2, random_state=23)

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)
print(classification_report(y_test, previsoes))

#Gera a matrix de confusão
confusion_matrix(y_test, previsoes)
cm = ConfusionMatrix(modelo)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)

#Plotando a árvore
tree.plot_tree(Y)
plt.show()