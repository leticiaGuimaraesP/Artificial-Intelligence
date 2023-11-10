import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Leitura dos dados
data = pd.read_csv('breast-cancer.csv');

# Pré-processamento
X = data.drop('Class', axis=1)
y = data['Class']
X = pd.get_dummies(X)

# Conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento da rede neural criada
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

predictions = mlp.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia: {accuracy:.2f}')