# getting required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('../../titanic_data.csv')

#singling out AGE attribute
train_data = train_data[['Age','Survived']]

#check for null values and drop them
##train_data.info()

train_data = train_data.dropna()

fig1 = plt.figure()
plt.hist(x=[np.array(train_data[train_data['Survived']==0]['Age']),np.array(train_data[train_data['Survived']==1]['Age'])],bins = 20,stacked=True, label=['Died','Survived'])
plt.xlabel('Age')
plt.ylabel('No of passengers')
plt.title('Stacked Histogram on Age')
plt.legend()
#plt.show()
plt.savefig('plots/Age_survival_distribution.png',bbox_inches='tight')

# separating features and labels
X = train_data.drop('Survived',axis=1)
Y = train_data['Survived']

## Train using the most common models
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X, Y)
acc_log = round(logreg.score(X, Y) * 100, 2)

# SVC
svc = SVC()
svc.fit(X, Y)
acc_svc = round(svc.score(X, Y) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X, Y)
acc_perceptron = round(perceptron.score(X, Y) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, Y)
acc_decision_tree = round(decision_tree.score(X, Y) * 100, 2)

models = pd.DataFrame({
    'Model': ['SVM', 'Logistic Regr.','Perceptron','Decision Tree'],
    'Score': [acc_svc, acc_log, acc_perceptron, acc_decision_tree]})

models = models.sort_values(by='Score', ascending=True)
models = models.reset_index(drop=True)
print models

l = ['{}\n{}'.format(models['Model'][i],models['Score'][i]) for i in range(len(models))]

fig2 = plt.figure()
plt.bar(np.arange(len(models)),models['Score'],alpha=0.5,tick_label = l,width = 0.4)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accruacy of classifiers trained on AGE')
plt.savefig('plots/Accuracy_classifier.png',bbox_inches='tight')
