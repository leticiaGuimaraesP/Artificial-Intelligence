import pandas as pd
from chefboost import Chefboost as chef

df = pd.read_csv('Lista3/risco.csv')
df

config = {'algorithm': 'C4.5'}
model = chef.fit(df, config=config, target_label = 'Risco' )

#Ruim,Alta,Nenhuma,25000,Alto
#prediction = chef.predict(model, param = ['Ruim', 'Alto', 'Nenhum', '25000'])

config = {'algorithm': 'ID3'}
model = chef.fit(df, config=config, target_label = 'Risco' )

config = {'algorithm': 'CART'}
model = chef.fit(df, config = config, target_label = 'Risco')