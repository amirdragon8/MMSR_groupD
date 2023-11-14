#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


# ## Uploading datasets

# In[2]:


df_names = pd.read_table('id_information_mmsr.tsv')
df_names.head(5)


# In[3]:


df_bert = pd.read_table('id_lyrics_bert_mmsr.tsv')
df_bert.head(5)


# In[4]:


df_word2vec = pd.read_table('id_lyrics_word2vec_mmsr.tsv')
df_word2vec.head(5)


# In[5]:


df_tfidf = pd.read_table('id_lyrics_tf-idf_mmsr.tsv')
df_tfidf.head(5)


# ## 3 chosen songs

# In[6]:


song1 = df_names[df_names['id'] == 'ziT77Si01mOb5oZg']; song1


# In[7]:


song2 = df_names[df_names['id'] == 'cVZd1wCtRYIqRnaV']; song2


# In[8]:


song3 = df_names[df_names['id'] == 'h0Jaex0Pdbn3aVXv']; song3


# ## Function returns artist and song name with given id

# In[9]:


def song_info(id_):
    return df_names[df_names['id'] == id_]


# ## Task 1. Random Baseline.
# 
# Regardless of the query track, this retrieval system randomly selects N tracks from the rest of the catalog. Make sure that the system produces new results for each query / run.

# In[10]:


def random_baseline(song):
    song_id = song['id'].values[0]
    return df_names.loc[(df_names['id'] != song_id)].sample(n=10)


# In[11]:


random_baseline(song1)


# In[12]:


random_baseline(song2)


# In[13]:


random_baseline(song3)


# ## Task 2. Text-based(cos-sim, tf-idf).
# 
# Given a query, this retrieval system selects the N tracks that are most similar to the query track. The similarity is measured as cosine similarity between the tf-idf representations of the lyrics of the tracks. I.e.
# 
# **𝑠𝑖𝑚(𝑞𝑢𝑒𝑟𝑦, 𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘) = 𝑐𝑜𝑠(𝑡𝑓_𝑖𝑑𝑓(𝑞𝑢𝑒𝑟𝑦), 𝑡𝑓_𝑖𝑑𝑓(𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘))**
# 

# In[14]:


def cos_sim_tfidf(song):
    song_id = song['id'].values[0]
    song_vec = df_tfidf.loc[df_tfidf['id'] == song_id] #target song vector
    df_temp = df_tfidf.loc[(df_tfidf['id'] != song_id)].copy() #make copy id because we want to add new column later, this dataset without target song
    cosine_sim = cosine_similarity(df_temp.iloc[:, 1:], song_vec.iloc[:, 1:]) #similarity between songs from dataset and target song
    df_temp['cos_sim'] = cosine_sim #add column with counted similarity
    ids = df_temp.sort_values(by='cos_sim', ascending=False).head(10)[['id', 'cos_sim']] #take 10 the greatest similarity values, we need id and similarity value
    result = pd.merge(ids, df_names, how='left', on='id') #merge table to represent names of found songs
    return result


# In[15]:


cos_sim_tfidf(song1)


# In[16]:


cos_sim_tfidf(song2)


# In[17]:


cos_sim_tfidf(song3)


# ## Task 3. Text-based(cos-sim, \<feature>)
# 
# Similar to Text-based(cos-sim, tf-idf), however choose a different text-based feature instead of tf-idf (i.e., word2vec or BERT representations of the lyrics)
# 
# **𝑠𝑖𝑚(𝑞𝑢𝑒𝑟𝑦, 𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘) = 𝑐𝑜𝑠(< 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 > (𝑞𝑢𝑒𝑟𝑦), < 𝑓𝑒𝑎𝑡𝑢𝑟𝑒 > (𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘))**

# In[18]:


def cos_sim_bert(song):
    song_id = song['id'].values[0]
    song_vec = df_bert.loc[df_bert['id'] == song_id] #target song vector
    df_temp = df_bert.loc[(df_bert['id'] != song_id)].copy() #make copy id because we want to add new column later, this dataset without target song
    cosine_sim = cosine_similarity(df_temp.iloc[:, 1:], song_vec.iloc[:, 1:]) #similarity between songs from dataset and target song
    df_temp['cos_sim'] = cosine_sim #add column with counted similarity
    ids = df_temp.sort_values(by='cos_sim', ascending=False).head(10)[['id', 'cos_sim']] #take 10 the greatest similarity values, we need id and similarity value
    result = pd.merge(ids, df_names, how='left', on='id') #merge table to represent names of found songs
    return result


# In[19]:


cos_sim_bert(song1)


# In[20]:


cos_sim_bert(song2)


# In[21]:


cos_sim_bert(song3)


# ## Task 4. Text-based(\<similarity>, \<feature>)
# 
# Similar to Text-based(cos-sim, <feature>), however choose a new combination of similarity measure and text-based feature (e.g., you can use cos-sim with a representation of the lyrics not selected for previous systems yet)
# 
#     
# **𝑠𝑖𝑚(𝑞𝑢𝑒𝑟𝑦, 𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘) = <𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦> (<𝑓𝑒𝑎𝑡𝑢𝑟𝑒>(𝑞𝑢𝑒𝑟𝑦), <𝑓𝑒𝑎𝑡𝑢𝑟𝑒>(𝑡𝑎𝑟𝑔𝑒𝑡_𝑡𝑟𝑎𝑐𝑘))**

# In[22]:


def euc_sim(song):
    song_id = song['id'].values[0]
    song_vec = df_word2vec.loc[df_word2vec['id'] == song_id] #target song vector
    df_temp = df_word2vec.loc[(df_word2vec['id'] != song_id)].copy() #make copy id because we want to add new column later, this dataset without target song
    euc_sim = euclidean_distances(df_temp.iloc[:, 1:], song_vec.iloc[:, 1:]) #similarity between songs from dataset and target song
    df_temp['euc_sim'] = euc_sim #add column with counted similarity
    ids = df_temp.sort_values(by='euc_sim', ascending=True).head(10)[['id', 'euc_sim']] #take 10 the greatest similarity values, we need id and similarity value
    result = pd.merge(ids, df_names, how='left', on='id') #merge table to represent names of found songs
    return result


# In[23]:


euc_sim(song1)


# In[24]:


euc_sim(song2)


# In[25]:


euc_sim(song3)

