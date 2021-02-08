#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel

class BetterRnnlm:
    def __init__(self, vocab_size = 10000, wordvec_size = 650, hidden_size = 650, dropout_ratio = 0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V,D)/100).astype('f')
        lstm_Wx1 = (rn(D, 4*H)/np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4*H).astype('f')
        lstm_Wx2 = (rn(D, 4*H)/np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_wh1, lstm_b1, stateful = True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_wh1, lstm_b1, stateful = True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2],self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        
        self.params, self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, xs, train_flg = False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts, train_flg = True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()


# In[ ]:


import sys
sys.path.append('..')
from common import config
# config.GPU = True
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util improt eval_perplexity
from dataset import ptb
# from better_rnnlm import BetterRnnlm

batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch = 1, batch_size = batch_size, time_size = time_size, max_grad = max_grad)
    
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('검증 퍼플렉서티: ', ppl)
    
    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr
    
    model.reset_state()
    print('-'*50)


trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval = 20)
trainer.plot(ylim=(0,500))

model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("테스트 퍼플렉서티: ", ppl_test)

model.save_params()

