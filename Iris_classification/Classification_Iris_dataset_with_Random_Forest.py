#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# carraga os dados 
data = pd.read_csv('Iris.csv')
# exibe as 5 primeiras linhas
data.head()


# In[3]:


# verifica quais são as espécies
data['Species'].value_counts()


# In[4]:


# transformas especies em valores númericos para aplicar a classificação
# função que transforma os dados 
from sklearn.preprocessing import LabelEncoder 
# instancia a funcao
label_enconder = LabelEncoder()
#aplica a função e cria uma nova coluna com os dados
data['Species_numeric'] = label_enconder.fit_transform(data['Species'])
data.head()


# In[5]:


# verifica os códigos atribuidos a cada espécie
data['Species_numeric'].value_counts()


# In[6]:


print(data.corr())
sns.heatmap(data.corr(),cbar=True)


# In[7]:


data.drop('Id',axis='columns', inplace=True)


# In[8]:


print(data.corr())
sns.heatmap(data.corr(),cbar=True,cmap="winter_r")


# In[9]:


# vamos ver os nomes das colunas
data.columns.values


# In[12]:


#Separa os atributos da classe
X = data.filter(items=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
y = data.filter(items=['Species_numeric'])


# In[13]:


X.shape


# In[14]:


y.shape


# In[29]:


#Separa em treino e teste
from sklearn.model_selection import train_test_split
X_treino,X_test,y_treino,y_test = train_test_split(X,y,test_size =0.3,random_state=40)


# In[30]:


# Aplicando nosso modelo de classificação 
from sklearn.ensemble import RandomForestClassifier
modelo_rf = RandomForestClassifier(n_estimators=150,criterion='entropy',random_state=40) #rf = random forest


# In[31]:


#aplicando o modelo 
modelo_rf.fit(X_treino,y_treino)


# In[33]:


#eficácia do modelo
modelo_rf.score(X_test,y_test)*100


# In[ ]:




