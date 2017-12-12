# getting required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# read in data
train_data = pd.read_csv('../../titanic_data.csv')

# drop null values and convert alphanumeric to numeric values
train_data = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Survived']]
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data = train_data.dropna()

# separating features and labels
X = train_data.drop('Survived',axis=1)
Y = train_data['Survived']

## Train using Logistic Regression
logreg = LogisticRegression()
logreg.fit(X, Y)

# normalise the coefficents and take absolute value
imp = (np.std(X)*(logreg.coef_[0])).abs()
imp = round(100*imp/sum(imp),1)
imp.sort_values(ascending = False, inplace=True)
print imp

plt.figure()
plt.bar([i*.5 for i in range(len(imp))],imp,width=0.4,tick_label = imp.index)
plt.xlabel('Attribute')
plt.ylabel('%tage importance')
plt.title('Relative Importance of Features')
plt.savefig('plots/relative_feature_importance.png',bbox_inches='tight')
