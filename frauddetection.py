import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('creditcard.csv')
data=data.sample(frac=0.1,random_state=1)
print(data.columns)
print(data.describe())

#Data Analysis
data.hist(figsize=(20,20))
plt.show()

Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
print(len(Fraud))
print(len(Valid))
outlier_fraction=len(Fraud)/len(Valid)
print(len(Fraud)/len(Valid))

#Correlation Matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

#Data Preprocessing
columns=data.columns.tolist()
columns=[c for c in columns if c not in ['Class']]
target='Class'
X=data[columns]
y=data[target]

#Building Models
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define random state
state=1
#define outlier detection methods
classifiers={
        'Isolation Forest': IsolationForest(max_samples=len(X),contamination=outlier_fraction,
                                            random_state=state),
        'Local Outlier Factor':LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)                                    
}

#fit the model
n_outliers=len(Fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name=='Local Outlier Factor':
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
    #Reshape prediction values 0 for inliers(valid) and 1 for outliers(fraud)
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred!=y).sum()
    #Run classification metrics
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))    


