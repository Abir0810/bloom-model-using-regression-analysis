#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd


# In[71]:


df=pd.read_csv(r"C:\Users\Abir.DESKTOP-U8C2SSO\Downloads\Tulip database.csv")


# In[72]:


df.head(2)


# In[73]:


from sklearn import linear_model


# In[74]:


from sklearn.linear_model import LinearRegression


# In[75]:


reg=LinearRegression()


# In[76]:


y = df[['Bloom']]
x = df.drop(['Bloom'],axis=1)
x=x.dropna()


# In[77]:


x.head(2)


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=10)


# In[80]:


from sklearn.linear_model import LinearRegression


# In[81]:


logmodel=LinearRegression()


# In[82]:


logmodel.fit(x_train, y_train)


# In[83]:


x_train


# In[84]:


logmodel.predict([[24,26,1,1,40,1,1,1,1]])


# In[85]:


logmodel.score(x_test,y_test)


# In[86]:


logmodel.coef_


# In[87]:


logmodel.intercept_


# In[88]:


from sklearn.linear_model import Ridge


# In[89]:


log = Ridge(alpha=1.0)


# In[90]:


log.fit(x_train,y_train)


# In[91]:


log.predict([[24,26,1,1,40,1,1,1,1]])


# In[92]:


log.score(x_test,y_test)


# In[93]:


log.coef_


# In[97]:


log.intercept_


# In[98]:


from sklearn import linear_model


# In[99]:


from sklearn.linear_model import Lasso


# In[100]:


clf = linear_model.Lasso(alpha=0.1)


# In[101]:


clf.fit(x_train, y_train)


# In[102]:


clf.predict([[24,26,1,1,40,1,1,1,1]])


# In[103]:


clf.score(x_test,y_test)


# In[104]:


ft = Lasso(alpha=1).fit(x_train, y_train)
print(ft.intercept_)
ft = LinearRegression().fit(x_train, y_train)
print(ft.intercept_)


# In[105]:


from sklearn.gaussian_process import GaussianProcessRegressor


# In[106]:


gpr = GaussianProcessRegressor()


# In[107]:


gpr.fit(x_train,y_train)


# In[108]:


gpr.fit(x_train, y_train)


# In[109]:


gpr.predict([[24,26,1,1,40,1,1,1,1]])


# In[110]:


gpr.score(x_test,y_test)


# In[118]:


from sklearn.neural_network import MLPClassifier


# In[119]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# In[120]:


clf.fit(x_train,y_train)


# In[121]:


clf.predict([[24,26,1,1,40,1,1,1,1]])


# In[122]:


clf.score(x_test,y_test)


# In[13]:


a=['Lasso','GaussianProcess','Neural Network','Ridge','Linear']


# In[14]:


b=['17','36','79','82','84']


# In[15]:


from matplotlib import pyplot as plt 


# In[16]:


plt.bar(b,a)
plt.ylabel("Algorithm")
plt.xlabel("Accuracy %")
plt.title("Algorithm Accuracy Comparison")
plt.show()







