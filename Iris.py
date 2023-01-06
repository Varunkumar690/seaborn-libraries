#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


iris=sns.load_dataset("iris")


# In[4]:


iris


# In[5]:


iris.describe()


# In[6]:


iris.head()


# In[7]:


iris.tail()


# In[8]:


iris.info()


# In[9]:


iris.isnull().sum()


# In[10]:


print(np.unique(iris['sepal_length']))


# In[11]:


iris.columns


# In[14]:


iris.dropna(axis=0,subset=["species"],inplace=True)
y = iris.species
x=iris.drop(['species'],axis=1).select_dtypes(exclude=["int64"])


# In[15]:


fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20,5))
index=0
ax=ax.flatten()
for col, value in iris.items():
    sns.boxplot(y = iris[col], x=iris['sepal_length'], data = iris, ax=ax[index])
    index += 1
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[16]:


sns.pairplot(iris)


# In[17]:


fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20,10))
index=0
ax=ax.flatten()
for col, value in iris.items():
    sns.histplot(value, ax=ax[index])
    index += 1
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[18]:


fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20,5))
index=0
ax=ax.flatten()
for col, value in iris.items():
    sns.countplot(value, ax=ax[index])
    index += 1
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[19]:


iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map(sns.scatterplot)


# In[20]:


g = sns.PairGrid(iris)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[21]:


g = sns.PairGrid(iris, hue="species")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()


# In[22]:


g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
g.map(sns.scatterplot)


# In[23]:


g = sns.PairGrid(iris)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)


# In[25]:


sns.pairplot(iris, hue="species", height=2.5)


# In[26]:


g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", height=2.5)


# In[ ]:





# In[ ]:




