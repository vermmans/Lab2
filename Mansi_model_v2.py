#!/usr/bin/env python
# coding: utf-8

# In[13]:


# import libraries
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

pd.set_option('display.max_columns', None)


# In[14]:


# read the data
df = pd.read_csv('Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data', header=None, 
                 names=['id', 'diagnosis', 'radius_mean', 'texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'])


# In[15]:


X = df.drop('diagnosis',axis = 1)
y = df['diagnosis']


# In[16]:


#splitting our data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[17]:


#scaling the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# In[18]:


#trying logistic Regression model
from sklearn.linear_model import LogisticRegression
LogR = LogisticRegression()
LogR.fit(X_train, y_train)


# In[19]:


# Predicting the target variable 'y_pred' using the trained logistic regression model 'LogR' on the test data 'X_test'.
y_pred = LogR.predict(X_test)


# In[20]:


# Calculate and print the accuracy score by comparing the actual test labels (y_test) with the predicted labels (y_pred)
print(accuracy_score(y_test, y_pred))


# In[22]:


# Define the filename for saving the trained logistic regression model
filename = 'mansi_model_v2.sav'
# save the trained logistic regression model 'rfc' to a file using pickle
pickle.dump(LogR, open(filename, 'wb'))

