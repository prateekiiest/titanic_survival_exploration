import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('titanic_data.csv')

comb = pd.DataFrame([data[data['Survived']==0]['Embarked'].value_counts(),
	data[data['Survived']==1]['Embarked'].value_counts()])
comb.index = ['Dead','Survived']
comb.plot(kind = 'bar',stacked = True)
