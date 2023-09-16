import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Criando o dataframe de exemplo
df = pd.read_csv('Lista4/jogar.csv')

X_prev = df.iloc[:, 0:4].values #Seleciona todas as linhas do DataFrame e extrai as colunas de 0 até 9
y_classe = df.iloc[:, 4].values 

#lb = LabelEncoder()
#X_prev[:,0] = lb.fit_transform(X_prev[:,0]) 
#X_prev[:,1] = lb.fit_transform(X_prev[:,1]) 
#X_prev[:,2] = lb.fit_transform(X_prev[:,2]) 
#X_prev[:,3] = lb.fit_transform(X_prev[:,3])

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2, 3])], remainder='passthrough') #Determina a coluna que o OneHotEncoder será aplicado
X_prev = onehotencoder_restaurante.fit_transform(X_prev) #Aplica o ColumnTransformer aos dados de entrada

#X_train, X_test, y_train, y_test = train_test_split(X_prev, y_classe, test_size=0.2, random_state=42)

# Criando o modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Treinando o modelo
modelo.fit(X_prev, y_classe)

nova_instancia = [[0, 1, 0, 0, 0, 1, 1, 0, 1, 0]]
y_pred = modelo.predict(nova_instancia)
print('Aparência = Nublado; Temperatura = Quente; Umidade = Alta; Ventando = Não')
print(y_pred)

nova_instancia = [[1, 0, 0, 0, 1, 0, 0, 1, 0, 1]]
y_pred = modelo.predict(nova_instancia)
print('Aparência = Chuva; Temperatura = Fria; Umidade = Normal; Ventando = Sim')
print(y_pred)
