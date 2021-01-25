#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello. '
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size = 1)
W = ppmi(C)

#SVD
U, S, V = np.linalg.svd(W)


# In[2]:


for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0], U[word_id, 1]))
    
plt.scatter(U[:,0], U[:,1], alpha = 0.5)
plt.show()


# In[ ]:




