import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from six.moves import cPickle
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
data=pd.read_csv("pp-complete.csv",header=None,skiprows=5000,nrows=5000)
n=[2,4,6,11,1]
data_reduced=np.array(data.iloc[:,n])
enc.fit(data_reduced[:,1:3])
#data generator
def data_processor(m,n):
# loading:
    for i in range (m,n):
        data=pd.read_csv("pp-complete.csv",header=None,skiprows=m*5000,nrows=5000)
        n=[2,4,6,11,1]
        data_reduced=np.array(data.iloc[:,n])
# one hot encoding:
        data_onehot=enc.transform(data_reduced[:,1:3])
        data_reduced[np.where(data_reduced[:,0]<'2016-01-01 00:00')[0],0]=0
        data_reduced[np.where(data_reduced[:,0]!=0)[0],0]=1
        data_reduced[np.where(data_reduced[:,3]=='LONDON')[0],3]=1
        data_reduced[np.where(data_reduced[:,3]!=1)[0],3]=0
        data_onehot=np.hstack((data_reduced[:,0].reshape(5000,1),data_onehot,data_reduced[:,3].reshape(5000,1)))
#stacking 5000 rows into 25000 matrices:
        if i==m:
            data_p=data_onehot
            ans=np.int8(data_reduced[:,4])
        else:
            data_p=np.vstack((data_p,data_onehot))
            ans=np.vstack((ans.reshape(ans.shape[0],1),np.int8(data_reduced[:,4]).reshape(5000,1)))
    return data_p, ans

#training process
def main():
    m=0
    err=np.zeros(100)
    for i in range (0,100):
# data generation:
        data,ans=data_processor(i*50,i*50+50)
# cross validation sets:
        x_train,x_test,y_train,y_test=train_test_split(data[np.where(data[:,0]==0)[0],:],ans[np.where(data[:,0]==0)[0]],test_size=0.2)
# define algorithms:
        rf =  RandomForestClassifier(n_estimators=10)
        df_x_train = x_train[:,1::]
        rf.fit(df_x_train,y_train)
# save single model:
        with open('rf'+str(i)+'.pkl', 'wb') as f:
            cPickle.dump(rf, f)
        df_x_test = x_test[:,1::]
        if i>0:
# load model and make prediction:
            for k in range (m,i):
                with open('rf'+str(k)+'.pkl', 'rb') as f:
                    rf = cPickle.load(f)
                pred = rf.predict(df_x_test)
                if k==0:
                    y_pred=pred
                else:
                    y_pred+=pred
                err[k]=mean_squared_error(y_pred/(k+1),y_test)
# termination
                if k>5:
                    if err[k]<=np.min(err[k-5:k-1]):
                        print(k-3)
                        break
                        break
            m=i
    print(k-3)