import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
binwidth=0.2
train=pd.read_csv("/home/vishal/Desktop/train(1).csv")
train_data=train[['Sex','Survived']]
d={"female":0,"male":1}
try:
    train_data['Sex']=train_data['Sex'].apply(lambda x:d[x])
except:
    print("There are some NaN in the input")
plt.hist(train_data.loc[train['Survived']==1,'Sex'],bins=[d['female'],d['female']+binwidth,d['male']-binwidth,d['male']])
plt.title('Passengers Survived Histogram wrt Sex')
plt.xlabel("Sex: 0 for female,1 for male")
plt.ylabel("No of passengers Survived")
plt.show()

#Seperating the features and the labels
X=train_data.drop('Survived',axis=1)
Y=train_data['Survived']

#Train by using Decision tree and random forests

decision_tree=DecisionTreeClassifier()
decision_tree.fit(X,Y)
print("Accuracy of Decision Tree=",round(decision_tree.score(X,Y)*100,2))

random_forest=RandomForestClassifier()
random_forest.fit(X,Y)
print("Accuracy of random forest is ",round(random_forest.score(X,Y)*100,2))
