import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('whitegrid')

train_titanic = pd.read_csv('train.csv')

#print(train_titanic.head())

#cabin must be removed or can be changed to cabin known, 1 or 0
#sns.heatmap(train_titanic.isnull(), yticklabels = False, cbar = False)

#most ppl who survived were women
#sns.countplot(x = 'Survived', hue= 'Sex', data= train_titanic)

#most ppl that did not survive is from class 3
#sns.countplot(x = 'Survived', hue= 'Pclass', data= train_titanic)

#most passengers are younger, towards age 10-30
#sns.distplot(train_titanic['Age'].dropna(), kde = False, bins = 30)

#most ppl do not have sibling or spouse. The 1 may represent spouse
#sns.countplot(x = 'SibSp', data=train_titanic)

#most fares are on the cheaper side, 0 -50. Most passengers are 3rd class
#train_titanic['Fare'].hist(bins = 40)

#most ppl on class 1 and 2 are older while in class 3 are younger
#sns.boxplot(x= 'Pclass', y = 'Age', data = train_titanic)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train_titanic['Age'] = train_titanic[['Age', 'Pclass']].apply(impute_age, axis = 1)

#sns.heatmap(train_titanic.isnull(), yticklabels= False, cbar = False)

train_titanic.drop('Cabin', axis = 1, inplace = True)

train_titanic.dropna(inplace = True)
 
#sns.heatmap(train_titanic.isnull(), yticklabels= False, cbar = False)

sex = pd.get_dummies(train_titanic['Sex'], drop_first = True)

embark = pd.get_dummies(train_titanic['Embarked'], drop_first = True)

classes = pd.get_dummies(train_titanic['Pclass'], drop_first = True)

train_titanic = pd.concat([train_titanic, sex, embark, classes], axis = 1)

train_titanic.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'Pclass'], axis = 1, inplace = True)
train_titanic.drop(['PassengerId'], axis = 1, inplace = True)

y = train_titanic['Survived']
X = train_titanic.drop('Survived', axis = 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 149)


regressor = LogisticRegression(solver = 'lbfgs')
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)

print(classification_report(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))










































































































