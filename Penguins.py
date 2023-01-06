#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df=sns.load_dataset("penguins")


# In[16]:


df


# In[17]:


df.head(20)


# In[18]:


df.tail(20)


# In[19]:


df.isnull()


# In[20]:


df.isnull().sum()


# In[21]:


df.describe()


# In[22]:


penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")


# In[23]:


sns.pairplot(data=penguins, hue="species")


# In[24]:


penguins = sns.load_dataset("penguins")
sns.displot(penguins, x="flipper_length_mm")


# In[25]:


sns.displot(penguins, x="flipper_length_mm", binwidth=3)


# In[26]:


sns.displot(penguins, x="flipper_length_mm", bins=20)


# In[27]:


sns.displot(penguins, x="flipper_length_mm", hue="species")


# In[28]:


sns.displot(penguins, x="flipper_length_mm", hue="species", element="step")


# In[29]:


sns.displot(penguins, x="flipper_length_mm", hue="species", multiple="stack")


# In[32]:


sns.displot(penguins, x="flipper_length_mm", hue="sex", multiple="dodge")


# In[33]:


sns.displot(penguins, x="flipper_length_mm", col="sex")


# In[34]:


sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density")


# In[35]:


sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density", common_norm=False)


# In[36]:


sns.displot(penguins, x="flipper_length_mm", hue="species", stat="probability")


# In[37]:


sns.displot(penguins, x="flipper_length_mm", kind="kde")


# In[38]:


sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=.25)


# In[39]:


sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=2)


# In[40]:


sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")


# In[41]:


sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", multiple="stack")


# In[42]:


sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)


# In[43]:


sns.displot(penguins, x="flipper_length_mm", kind="ecdf")


# In[44]:


sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")


# In[45]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")


# In[46]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")


# In[47]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")


# In[48]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")


# In[49]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5))


# In[50]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5), cbar=True)


# In[51]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", thresh=.2, levels=4)


# In[52]:


sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", levels=[.01, .05, .1, .8])


# In[53]:


sns.jointplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    kind="kde")


# In[54]:


g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)


# In[55]:


sns.displot(
    penguins, x="bill_length_mm", y="bill_depth_mm",
    kind="kde", rug=True
)


# In[56]:


sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
sns.rugplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")


# In[57]:


sns.pairplot(penguins)


# In[58]:


g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)

