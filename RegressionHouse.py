
# coding: utf-8

# In[ ]:


import torch
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None) 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('whitegrid') 
import warnings
warnings.filterwarnings('ignore') 
import datetime
import json
import logging
import os
import shutil
from scipy import stats
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors 
from sklearn.metrics import mean_squared_error 
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeRegressor
from math import log
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def Dec_scale(df):
    for x in df:
        p = df[x].astype("int").max()
        q = len(str(abs(p)))        
        df[x] = df[x]/(10**q) 


# In[ ]:


def change2categorical(df,names):
  for x in names:
    df[x]=df[x].astype('object')
  


# In[ ]:



def save_checkpoint(state, is_best, checkpoint,time):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint,str(time)+ 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,str(time)+ 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


# In[ ]:


df=pd.read_csv("./drive/My Drive/DataSet/kc_house_data.csv",parse_dates=['date']);
df=df.drop(['id'],axis=1)


# In[ ]:


df.shape


# In[ ]:


df.head(5)


# In[ ]:


features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','grade','sqft_above','sqft_basement',
            'yr_built','yr_renovated','zipcode','sqft_living15','sqft_lot15']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="BuGn_r", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})


# In[ ]:


sns.boxplot(y=df["price"])
plt.show()


# In[ ]:


sns.pairplot(df)
plt.show()


# In[ ]:


df.describe()


# In[ ]:


missing=pd.DataFrame({'Missing count':df.isnull().sum()})
missing.plot.bar()


# In[13]:


count=0
for i in df:
  if str(i)=='date':
    continue
  plt.figure(count)
  plt.hist(df[i],alpha=0.7, rwidth=0.85)
  plt.tight_layout()
  plt.xlabel(i)
  plt.ylabel('Count')
  count=count+1
plt.show()


# In[ ]:


#preprocess data
data=pd.read_csv("./drive/My Drive/DataSet/kc_house_data.csv",parse_dates=['date'],dtype = np.float32);
data=data.drop(['id','date'],axis=1)

#One Hot Encoding for yr_renovated
data['yr_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

#remove outlier
housing=data

IQR = housing['price'].quantile(.75) - housing['price'].quantile(.25)
upper_bound = housing['price'].quantile(.75) + 3 * IQR
upper_bound_mask = housing.price > upper_bound

lower_bound = housing['price'].quantile(.25) - 3 * IQR
lower_bound_mask = housing.price < upper_bound
housing_no_outliers = housing[housing["price"] < upper_bound]
housing_no_outliers=housing_no_outliers[housing_no_outliers["price"]> lower_bound]
data=housing_no_outliers

#One Hot Encoding
change2categorical(data,['grade','view','condition','waterfront','floors','yr_renovated'])
data=pd.get_dummies(data)

#normalization by decimal scaling
Dec_scale(data)

X = data.drop('price', axis=1)
y=data["price"].astype('float32')
print(y.head(2))
y=y.values
X=X.astype('float32')

print(X.info())
print(X.head(10))
X=X.values


# In[ ]:


from sklearn.model_selection import train_test_split

features_train, features_test, targets_train, targets_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train)


featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test) 


train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

train_loader = torch.utils.data.DataLoader(train, batch_size =features_train.shape[0], shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size =features_test.shape[0]  , shuffle = False)


# In[ ]:


class ANNV2(nn.Module):
  def __init__(self):
    
    super(ANNV2, self).__init__()
    def layerReLU(inp,oup):
      return nn.Sequential(
          nn.Linear(inp,oup),
          nn.BatchNorm1d(oup),
          nn.ReLU(inplace=True),
          #nn.Dropout(0.7)
      )
    self.model = nn.Sequential(
        layerReLU(43,430),
        layerReLU(430,512),
        layerReLU(512,1024),
        layerReLU(1024,2048),
        layerReLU(2048,4096),
        layerReLU(4096,1024),
        layerReLU(1024,512),
        layerReLU(512,256),
        layerReLU(256,128),
    )
    self.fcout=nn.Linear(128,1)
    self.softplus4=nn.Softplus()
  def forward(self,x):
    out=self.model(x)
    out=self.fcout(out)
    
    out=self.softplus4(out)
    return out


# In[9]:


model=ANNV2()
model.cuda()
error = nn.L1Loss()
learning_rate =0.0025
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)
#load_checkpoint("./drive/My Drive/DataSet/ANNV1last.pth.tar", model, optimizer)
#for g in optimizer.param_groups:
#    g['lr'] = 0.001
count = 0
testloss=[]
loss_list = []
testloss_list=[]
iteration_list = []
accuracy_list = []
counts=[]
flag=0
while flag==0 :    
    for trains, labels in train_loader:        
        train = Variable(trains.float()).cuda()
        labels = Variable(labels.float()).view(-1,1).cuda()                
        optimizer.zero_grad()  
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()       
        count += 1
        if count % 1 == 0:                      
            for images, labels in test_loader:                
                test = Variable(images.float()).cuda()
                labels = Variable(labels.float()).view(-1,1).cuda()
                outputs = model(test)
                teloss=error(outputs, labels)
            loss_list.append(loss.data[0]*(10**7))           
            testloss_list.append(teloss.data[0]*(10**7))
            counts.append(count)
            iteration_list.append(count)
            if count%50==0:
              print('Smallest test : {}'.format(min(testloss_list)))
              print('Smallest train : {}'.format(min(loss_list)))                
              print('Iteration: {}  Loss: {}  Loss test: {} '.format(count, loss.data[0]*(10**7), teloss.data[0]*(10**7)))                
            if count % 500 == 0:
              plt.plot(range(0,len(loss_list)),loss_list)
              plt.plot(range(0,len(testloss_list)),testloss_list)
              plt.legend(['train','test'])
              plt.show()
              plt.pause(0.0001)
    if teloss.data<0.008:
      print('Iteration: {}  Loss: {}  Loss test: {} '.format(count, loss.data[0]*(10**7), teloss.data[0]*(10**7)))
      break          
plt.plot(range(0,len(loss_list)),loss_list)
plt.plot(range(0,len(testloss_list)),testloss_list)
plt.legend(['train','test'])
plt.show()
print("END")

