from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import sys



print("loading start")
train_X=pd.read_table('*************',index_col=0)

train_y=pd.read_table('*************',index_col=0)

#training data
print("train")
clf = RandomForestClassifier(n_estimators=1000,bootstrap=True,n_jobs=-1,random_state=0)
learn_time_s=time.time()
clf = clf.fit(train_X, train_y)
learn_time=time.time()-learn_time_s
filename = 'RF_model_15.sav'
#save model
pickle.dump(clf, open(filename, 'wb'))
print('learn time :%f[sec]'%learn_time)
