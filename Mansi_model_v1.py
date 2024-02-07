#!/usr/bin/env python
# coding: utf-8

# In[47]:


# import libraries
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

pd.set_option('display.max_columns', None)


# In[48]:


# read the data and the names of the feature used
df = pd.read_csv('Downloads/breast+cancer+wisconsin+diagnostic/wdbc.data', header=None, 
                 names=['id', 'diagnosis', 'radius_mean', 'texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'])


# In[49]:


df


# In[61]:


#Seperating data and labels
X = df.drop('diagnosis',axis = 1)
y = df['diagnosis']


# In[51]:


#lets see if there are any missing values
df.info()


# In[52]:


df


# In[53]:


#splitting our data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[54]:


#scaling the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# In[55]:


#trying random forest with as many as 50 decision trees!
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(random_state=20, n_estimators=50)
rfc.fit(X_train, y_train)


# In[56]:


# Predicting the target variable 'y_pred' using the trained RandomForestClassifier 'rfc' on the test data 'X_test'.
y_pred = rfc.predict(X_test)


# In[57]:


# Calculate and print the accuracy score by comparing the actual test labels (y_test) with the predicted labels (y_pred)
print(accuracy_score(y_test, y_pred))


# In[60]:


# Define the filename for saving the trained RandomForestClassifier
filename = 'mansi_model_v1.sav'
# save the trained logistic regression model 'rfc' to a file using pickle
pickle.dump(rfc, open(filename, 'wb'))

