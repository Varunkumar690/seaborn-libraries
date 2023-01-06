#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df=sns.load_dataset("tips")


# In[17]:


df


# In[18]:


df.sample(5)


# In[19]:


df.describe().T


# In[20]:


df.head()


# In[21]:


df.head(20)


# In[22]:


df.tail()


# In[23]:


df.tail(20)


# In[24]:


df.info()


# In[25]:


df.isnull()


# In[26]:


df.isnull().sum()


# In[27]:


df.corr()


# In[28]:


print(np.unique(df['tip']))


# In[29]:


df.columns


# In[30]:


fig, ax=plt.subplots(ncols=4, nrows=2, figsize=(20,10))
index=0
ax=ax.flatten()
for col, value in df.items():
    sns.ecdfplot(value, ax=ax[index])
    index+=1
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[31]:


sns.set(style="darkgrid")
sns.relplot(x = "total_bill", y = "tip", data = df);


# In[32]:


sns.relplot(x = "total_bill", y = "tip", hue= "smoker", data = df);


# In[33]:


sns.scatterplot(x = "total_bill", y = "tip", hue= "sex", data = df);


# In[34]:


sns.scatterplot(x = "total_bill", y = "tip", hue= "smoker", style= "smoker", data = df);


# In[73]:


sns.relplot(x = "total_bill",y = "tip",hue = "smoker",style = "time",
            height = 6,
            data = df)


# In[36]:


sns.relplot(x = "total_bill", y = "tip", hue = "size", height = 7, data = df);


# In[37]:


sns.relplot(x = "total_bill", y = "tip", size = "size", sizes = (20,100), hue = "size", data = df);


# In[38]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "time", data = df);


# In[39]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "sex", data = df);


# In[40]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "day", data = df);


# In[41]:


sns.catplot(x = "day", y = "total_bill", data = df);


# In[42]:


sns.catplot(x = "day", y = "total_bill", hue = "sex", data = df);


# In[43]:


sns.relplot(x = "total_bill", y = "tip", hue = "smoker", col = "day", data = df);


# In[44]:


sns.catplot(x = "day", y = "total_bill", data = df);


# In[45]:


sns.catplot(x = "day", y = "total_bill", hue = "sex", data = df);


# In[46]:


sns.catplot(x = "day", y = "total_bill", jitter = False, hue = "sex", alpha = .33, data = df);


# In[47]:


sns.swarmplot(x = "day", y = "total_bill", data = df);


# In[48]:


sns.swarmplot(x = "day", y = "total_bill", hue = "sex", alpha = .75, data = df);


# In[49]:


sns.swarmplot(x ="size", y = "total_bill", data = df);


# In[50]:


sns.swarmplot(x ="size", y = "total_bill", hue = "sex", alpha =.7, data = df);


# In[51]:


sns.catplot(x = "smoker", y = "tip", order = ["No", "Yes"], data = df);


# In[52]:


sns.boxplot(x = "day", y = "total_bill", data = df);


# In[53]:


sns.boxplot(x = "day", y = "total_bill", hue = "sex", data = df);


# In[54]:


sns.boxplot(x = "day", y = "total_bill", hue = "smoker", data = df);


# In[55]:


df["weekend"] = df["day"].isin(["Sat","Sun"])
df.sample(7)


# In[56]:


sns.boxplot(x = "day", y = "total_bill", hue = "weekend", data = df);


# In[57]:


sns.boxenplot(x= "sex", y = "tip", hue = "smoker", data = df);


# In[58]:


sns.violinplot(x ="day", y = "total_bill", hue = "time", data = df);


# In[59]:


sns.violinplot(x ="day", y = "total_bill", hue = "smoker", bw = .25, split = True, data = df);


# In[60]:


sns.violinplot(x="day", y="total_bill", hue="smoker", bw=.25, split=True, palette= "pastel", inner= "stick", data=df);


# In[61]:


sns.violinplot(x = "day", y = "total_bill", inner = None, data = df)
sns.swarmplot(x = "day", y = "total_bill", color = "k", size = 3, data = df);


# In[62]:


sns.barplot(x = "sex", y= "total_bill", hue = "smoker", data = df);


# In[63]:


sns.barplot(x = "day", y= "tip", hue = "smoker", palette = "ch:.25", data = df);


# In[64]:


sns.countplot(x = "day", hue ="sex", data = df)


# In[65]:


sns.countplot(x = "sex", hue = "smoker", palette = "ch:.25", data = df);


# In[66]:


sns.countplot(x = "day", hue = "size", palette = "ch:.25", data = df);


# In[67]:


sns.pointplot(x = "day", y = "tip", data= df);


# In[68]:


sns.pointplot(x = "day", y = "size", hue = "sex", linestyles = ["-", "--"], data= df);


# In[69]:


f, ax = plt.subplots(figsize = (7,3))
sns.countplot(x = "day", hue= "smoker", data = df)


# In[70]:


sns.catplot(x="day", y = "total_bill", hue = "smoker", col = "time", data = df);


# In[71]:


sns.catplot(x = "day", y = "total_bill", col = "sex", kind="box", data = df);

