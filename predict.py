from sklearn.metrics import accuracy_score
import pandas as pd
import time
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import sys

print("loading start")
test_X=pd.read_table(sys.argv[1],index_col=0,header=None)
test_y=pd.read_table(sys.argv[2],index_col=0,header=None)
#test data
clf= pickle.load(open('/home/sitoh/window/w15/RF_model_15.sav', 'rb'))
print("predict start")
pred_time_s=time.time()
pred=clf.predict(test_X)
pred_time=time.time()-pred_time_s


f_1=open('/home/sitoh/window/result/'+sys.argv[3]+'.result','wt')
f_1.write("predict time :"+str(pred_time)+"[sec]"+"\n")
f_1.write('Accuracy     :'+str(accuracy_score(test_y,pred))+"\n")
f_1.write('Precision    :'+str(precision_score(test_y,pred))+"\n")
f_1.write('Recall       :'+str(recall_score(test_y,pred))+"\n")
f_1.write('F score      :'+str(f1_score(test_y,pred))+"\n")
f_1.write('MCC          :'+str(matthews_corrcoef(test_y, pred))+"\n")
f_1.write("Futurie Importance\n")
fti = clf.feature_importances_
for j in fti:
        f_1.write(str(j)+"\n")



path='/home/sitoh/window/prob/'+sys.argv[3]+'.pro'
f = open(path,mode='wt')

f.write("data_no        0       1\n")
for vals,no in zip(clf.predict_proba(test_X),test_X.index):
        f.write(str(no)+"       "+str(vals[0])+"        "+str(vals[1])+"\n")

f.close()
f_1.close()
