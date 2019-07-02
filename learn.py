from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import sys



print("loading start")
train_X=pd.read_table('/home/hanbo/work/next_Deep/20181016_to_MBSJ2018/data/pssm/window15/train_data/train_train_w15_non_lbl2.fct',index_col=0)

train_y=pd.read_table('/home/hanbo/work/next_Deep/20181016_to_MBSJ2018/data/pssm/window15/train_data/train_train_w15_non_lbl2.lbl',index_col=0)

#pandas file->dataflame
#JOIN dataflame
#training data
#JOIN dataflame
print("train")
clf = RandomForestClassifier(n_estimators=1000,bootstrap=True,n_jobs=-1,random_state=0)
learn_time_s=time.time()
clf = clf.fit(train_X, train_y)
learn_time=time.time()-learn_time_s
filename = 'RF_model_15.sav'
pickle.dump(clf, open(filename, 'wb'))
print('learn time :%f[sec]'%learn_time)
