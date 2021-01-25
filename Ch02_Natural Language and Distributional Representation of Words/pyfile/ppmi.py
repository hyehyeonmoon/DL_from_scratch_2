#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def ppmi(C, verbose = False, eps = 1e-8):
    M = np.zeros_like(C, dtype = np.float32)
    N = np.sum(C)
    S = np.sum(C, axis = 0)
    total = C.shape[0] * C.shape[1]
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[j]*S[i])+eps)
            M[i,j] = max(0,pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total/100) == 0:
                    print("%.1f%% 완료" % (100*cnt/total))


# In[3]:


import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
print('동시발생 행렬')
print(C)
print('-'*50)
print('PPMI')
print(W)


# In[ ]:




