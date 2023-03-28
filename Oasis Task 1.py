#!/usr/bin/env python
# coding: utf-8
# # Oasis Infobyte ( Data Science Internship)
# 
# ## Task 1 : Performing Machine learning Model On IRIS Flower Dataset
# 
# ## Author - Patil Saloni Ravindra
# 
# ## March - P2 Batch Oasis Infobyte SIP

# ## Importing Libraries
# In[2]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
# In[3]:
data=pd.read_csv("C:/Users/admin/Documents/laptop/datasets/IRIS..csv")
data
# In[4]:
data=pd.read_csv("C:/Users/admin/Documents/laptop/datasets/IRIS..csv")
data
# In[5]:
data.tail()
# In[7]:
data.isnull().sum()
# In[8]:
data.shape
# In[9]:
data.dtypes
# In[10]:
data['Species'].unique()
# In[12]:
data.describe()
# # Data Visualization
# In[13]:
sns.pairplot(data)
# In[14]:
data.corr()
# In[15]:
sns.heatmap(data.corr(),annot=True)
plt.show()
# In[17]:
plt.boxplot(data['SepalLengthCm'])
plt.show()
# ### From above heatmap we can see that there is no outlier in the SepalLengthCm
# In[18]:
plt.boxplot(data['SepalWidthCm'])
plt.show()
# ### From the above boxplot we can see that there is some outlier predict in SepalWidthCm
# In[19]:
plt.boxplot(data['PetalLengthCm'])
plt.show()
# ### From the above boxplot we can see that there are some outlier predict in PetalLengthCm
# In[20]:
plt.boxplot(data['PetalWidthCm'])
plt.show()
# ### From the above boxplot we can see that there are some outlier predict in PetalWidthCm
# In[22]:
data.drop('Id',axis=1, inplace=True)
# In[23]:
spec={'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
data.Species=[spec[i] for i in data.Species]
data
# In[24]:
x=data.iloc[:,0:4]
x
# In[25]:
y=data.iloc[:,4]
y
# In[27]:
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
# # Training Model
# In[31]:
model=LinearRegression()
# In[32]:
model.fit(x,y)
# In[33]:
model.score(x,y)
# In[34]:
model.coef_
# In[35]:
model.intercept_
# # Making Predictions
# In[36]:
y_pred=model.predict(x_test)
# # Model Evolation 
# In[37]:
print("Mean Squared Error: %.2f" % np.mean((y_pred - y_test)**2))
# # Naive Bayes Algorithm
# In[41]:
from sklearn.naive_bayes import GaussianNB
accuracies={}
nb=GaussianNB()
nb.fit(x_train, y_train)
acc=nb.score(x_test, y_test)*100
accuracies['Naive Bayes']=acc
print("Accuracy of Naive Bayes:{:.2f}%".format(acc))
# In[42]:
nb.score(x_train,y_train)*100
# In[43]:
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
predictions = nb.predict(x_test)
predictions
sns.heatmap(confusion_matrix(y_test, predictions),annot = True)
plt.show()
# # Thank You!
# In[ ]:




