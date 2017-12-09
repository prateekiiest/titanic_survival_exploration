import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('titanic_data.csv')

emb_died = data[data['Survived']==0]['Embarked'].value_counts()
emb_survived = data[data['Survived']==1]['Embarked'].value_counts()
comb = pd.DataFrame([emb_died,emb_survived])
comb.index = ['Dead','Survived']
comb.plot(kind = 'bar',stacked = True)
