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


# In[2]:


df=sns.load_dataset("flights")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


sns.relplot(data=df, x="year", y="passengers", hue="month", kind="line")


# In[12]:


flights_wide = df.pivot(index="year", columns="month", values="passengers")
flights_wide.head()


# In[13]:


sns.relplot(data=flights_wide, kind="line")


# In[15]:


sns.relplot(data=df, x="month", y="passengers", hue="year", kind="line")


# In[16]:


sns.relplot(data=flights_wide.transpose(), kind="line")


# In[17]:


sns.catplot(data=flights_wide, kind="box")


# In[18]:


flights_dict = df.to_dict()
sns.relplot(data=flights_dict, x="year", y="passengers", hue="month", kind="line")


# In[19]:


flights_avg = df.groupby("year").mean()
sns.relplot(data=flights_avg, x="year", y="passengers", kind="line")


# In[20]:


year = flights_avg.index
passengers = flights_avg["passengers"]
sns.relplot(x=year, y=passengers, kind="line")


# In[21]:


sns.relplot(x=year.to_numpy(), y=passengers.to_list(), kind="line")


# In[22]:


flights_wide_list = [col for _, col in flights_wide.items()]
sns.relplot(data=flights_wide_list, kind="line")


# In[23]:


two_series = [flights_wide.loc[:1955, "Jan"], flights_wide.loc[1952:, "Aug"]]
sns.relplot(data=two_series, kind="line")


# In[24]:


two_arrays = [s.to_numpy() for s in two_series]
sns.relplot(data=two_arrays, kind="line")


# In[25]:


two_arrays_dict = {s.name: s.to_numpy() for s in two_series}
sns.relplot(data=two_arrays_dict, kind="line")


# In[26]:


flights_array = flights_wide.to_numpy()
sns.relplot(data=flights_array, kind="line")


# In[ ]:




