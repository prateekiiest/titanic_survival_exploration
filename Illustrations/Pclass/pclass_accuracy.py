# getting required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('../../titanic_data.csv')

#singling out Pclass attribute
train_data = train_data[['Pclass','Survived']]

#check for null values and drop them
##train_data.info()
#train_data = train_data.dropna()

table = train_data.groupby(['Survived','Pclass']).size().unstack()
print table

# plotting distribution of passenger classes
plt.figure(figsize=(3,3))
plt.pie([sum(table[i]) for i in table.columns],labels=['pclass1','pclass2','pclass3'])
plt.title('Distribution of Passenger Classes')
plt.savefig('plots/PassengerClass_distribution.png',bbox_inches='tight')

# plotting class distribution for each survival status
plt.figure()
plt.subplot(1,2,1)
plt.bar([0,1,2],table.iloc[1],width=0.5)
plt.xticks([0,1,2],(1,2,3))
plt.xlabel('Pclass')
plt.ylabel('No of passengers')
plt.title('Did not survive')

plt.subplot(1,2,2)
plt.bar([0,1,2],table.iloc[0],width=0.5)
plt.xticks([0,1,2],(1,2,3))
plt.xlabel('Pclass')
plt.ylabel('No of passengers')
plt.title('Survived')

plt.tight_layout()
plt.savefig('plots/PClass_v_survival_distribution.png',bbox_inches='tight')

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

plt.figure()
plt.bar(np.arange(len(models)),models['Score'],alpha=0.5,tick_label = l,width = 0.4)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Accruacy of classifiers trained on Pclass')
plt.savefig('plots/Accuracy_classifier.png',bbox_inches='tight')
