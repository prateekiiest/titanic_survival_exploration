import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

#Feature Engineering
df_train = pd.read_csv('titanic_data.csv')
df_test = pd.read_csv('test.csv')
targets = df_train.Survived
df_train.drop('Survived',axis = 1,inplace = True)
combined = df_train.append(df_test)

combined['Title'] = combined.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
combined.Title.value_counts()

Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
combined.Title = combined.Title.map(Title_Dictionary)

features_to_drop = ['PassengerId','SibSp','Parch']


gt_train = combined.head(891).drop(features_to_drop,axis = 1).groupby(['Sex','Pclass','Title']).median()
gt_test = combined.iloc[891:].drop(features_to_drop,axis = 1).groupby(['Sex','Pclass','Title']).median()


def fillAges(row, grouped_median):
 return (grouped_median.loc[row['Sex'],row['Pclass'],row['Title']]['Age']) 


combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, gt_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, gt_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)

combined['Fare'] = combined['Fare'].fillna(combined.Fare.mean())

combined['Ticket'] = combined.Ticket.apply(lambda t:t[0] if not t[0].isdigit() else 'X')

#Replace by mean for resulting Age nulls---effectively giving 3 outliers 
#but it should be negligible compared to 1309 values
combined['Age'] = combined['Age'].fillna(combined.Fare.mean())

combined['Embarked'] = combined['Embarked'].fillna('S')#Most occuring value
combined['FamilySize'] = combined['SibSp'] + combined['Parch']
combined = combined.drop(['PassengerId','Name','SibSp','Parch'],axis = 1)

combined.Cabin.fillna('U',inplace = True)
combined['Cabin'] = combined['Cabin'].map(lambda s:s[0])#Map each to first letter of cabin

combined['Sex'] = combined['Sex'].map({'male':0,'female':1})

# introducing other features based on the family size
combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

#encode all the categorical variables
features = ['Pclass','Sex','Ticket','Embarked','Cabin','Title']
for feature in features:
	df_encoded = pd.get_dummies(combined[feature], prefix=feature)
	combined = pd.concat([combined,df_encoded],axis = 1)
	combined.drop(feature,axis=1,inplace=True)

combined.to_csv('titanic_engineered.csv')

