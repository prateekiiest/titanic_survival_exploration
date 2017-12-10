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

gt_train = combined.head(891).drop(['PassengerId','SibSp','Parch'],axis = 1).groupby(['Sex','Pclass','Title']).median()
gt_test = combined.iloc[891:].drop(['PassengerId','SibSp','Parch'],axis = 1).groupby(['Sex','Pclass','Title']).median()


def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']
    
combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, gt_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, gt_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)

combined['Fare'] = combined['Fare'].fillna(combined.Fare.mean())

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

#Encode all variables
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
combined = pd.concat([combined,pclass_dummies],axis=1)
combined.drop('Pclass',axis=1,inplace=True)

#Encode all variables
s_dummies = pd.get_dummies(combined['Sex'], prefix="Sex")
combined = pd.concat([combined,s_dummies],axis=1)
combined.drop('Sex',axis=1,inplace=True)

#Encode all variables
t_dummies = pd.get_dummies(combined['Ticket'], prefix="Ticket")
combined = pd.concat([combined,t_dummies],axis=1)
combined.drop('Ticket',axis=1,inplace=True)


#Encode all variables
e_dummies = pd.get_dummies(combined['Embarked'], prefix="Embarked")
combined = pd.concat([combined,e_dummies],axis=1)
combined.drop('Embarked',axis=1,inplace=True)

#Encode all variables
c_dummies = pd.get_dummies(combined['Cabin'], prefix="Cabin")
combined = pd.concat([combined,c_dummies],axis=1)
combined.drop('Cabin',axis=1,inplace=True)

#Encode all variables
t_dummies = pd.get_dummies(combined['Title'], prefix="Title")
combined = pd.concat([combined,t_dummies],axis=1)
combined.drop('Title',axis=1,inplace=True)

combined.to_csv('titanic_engineered.csv')

