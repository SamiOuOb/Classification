import os

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
# %matplotlib inline
sns.set(font_scale=1.56)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
# loading data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_data = df_train.append(df_test)#合併train&test
df_data.reset_index(inplace=True, drop=True)#reset index(防止index重複問題)
# for display dataframe
from IPython.display import display
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
# ignore warning
import warnings
warnings.filterwarnings("ignore")

df_train.Age.fillna(df_train.Age.mean(),inplace=True) 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df_train['Sex'])
df_train['Sex']=le.transform(df_train['Sex'])
print(df_train.Sex)

features=['Pclass','Sex','Age']
X=df_train[features]
y=df_train['Survived']
dt=tree.DecisionTreeClassifier()
score=cross_val_score(dt,X,y,cv=5,scoring='accuracy')
import numpy as np
print(np.mean(score))

dt.fit(X,y)
test=pd.read_csv('test.csv')
test.Age.fillna(test.Age.mean(),inplace=True) 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(test['Sex'])
test['Sex']=le.transform(test['Sex'])
X_test=test[features]
result=dt.predict(X_test)
test.insert(1,'Survived',result)
final=test.iloc[:,0:2]

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": result
    })
submission.to_csv('submission.csv', index=False)