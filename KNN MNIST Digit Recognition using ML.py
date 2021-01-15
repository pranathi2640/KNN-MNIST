#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[12]:


X=dfx.values
Y=dfy.values

X=X[:,1:]
Y=Y[:,1:].reshape((-1,))

print(X)
print(X.shape)
print(Y.shape)


# In[13]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show


# In[14]:


query_x=np.array([2,3])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(query_x[0],query_x[1],color='red')
plt.show()


# In[17]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
    
def knn(X,Y,queryPoint,k=5):

    vals=[]
    
    m=X.shape[0]
    
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    vals=vals[:k]
    
    vals=np.array(vals)
    
   # print(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    
    return pred
    
    


# In[18]:


knn(X,Y,query_x)


# #MNIST Datasets

# In[19]:


df=pd.read_csv('train.csv')
print(df.shape)


# In[20]:


print(df.columns)


# In[21]:


df.head()


# In[22]:


#Create Numpy Array
data=df.values
print(data.shape)
print(type(data))


# In[23]:


X=data[:,1:]
Y=data[:,0]

print(X.shape,Y.shape)


# In[24]:


split=int(0.8*X.shape[0])
print(split)


# In[26]:


X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[28]:


def drawing(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()


# In[29]:


drawing(X_train[3])
print(Y_train[3])


# In[30]:


pred=knn(X_train,Y_train,X_test[0])
print(pred)


# In[33]:


drawing(X_test[9])
print(Y_test[9])


# In[ ]:




