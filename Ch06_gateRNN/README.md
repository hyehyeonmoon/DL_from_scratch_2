# Ch06_ 게이트가 추가된 RNN

생성일: Feb 8, 2021 12:02 AM
태그: 상현 김

기존의 RNN은 시계열 데이터의 장기 의존 관계를 학습하기 어렵습니다. 그 원인은 BPTT에서 기울기 소실 혹은 기울기 폭발이 일어나기 때문입니다.

기울기 폭발의 대책으로 기울기 클리핑(gradients clipping)을 사용합니다. 기울기 소실이 대책으로 gate를 사용한 RNN인 LSTM, GRU를 모델로 사용합니다.

LSTM 모델을 구현하고, ptb 데이터셋을 이용해 학습을 진행했습니다. 또한, LSTM 계층 다층화, 드랍아웃 계층 추가, 가중치 공유를 이용한 개선된 모델을 구현하고 학습을 진행했습니다.

**[정리]**

- 단순한 RNN의 학습에서는 기울기 소실과 기울기 폭발이 문제가 된다.
- 기울기 폭발에는 기울기 클리핑, 기울기 소실에는 게이트가 추가된 RNN(LSTM과 GRU 등)이 효과적이다.
- LSTM에는 input 게이트, forget 게이트, output 게이트 등 3개의 게이트가 있다.
- 게이트에는 전용 가중치가 있으며, 시그모이드 함수를 사용하여 0.0~1.0 사이의 실수를 출력한다.
- 언어 모델 개선에는 LSTM 계층 다층화, 드롭아웃, 가중치 공유 등의 기법이 효과적이다.
- RNN의 정규화는 중요한 주제이며, 드롭아웃 기반의 다양한 기법이 제안되고 있다.

**[파일 설명]**

- gradient_clipping.ipynb: 가중치 클리핑 구현한 파일 입니다.
- lstm.ipynb: LSTM을 구현한 파일 입니다.
- rnnlm.ipynb: LSTM 구조를 이용하여 RNNLM을 구현 및 학습한 파일 입니다.
- better_rnnlm.ipynb: 위의 rnnlm을 개선한 모델을 구현한 파일 입니다.

**[심화]**

## GRU(Gated Recurrent Unit)

![image](https://user-images.githubusercontent.com/68596881/107239168-f175b480-6a6b-11eb-801c-207af70c0fd6.png)

**r: reset gate**

**z: update gate**

![image](https://user-images.githubusercontent.com/68596881/107239192-f6d2ff00-6a6b-11eb-964c-4f1d89e66330.png)

reset 게이트는 과거의 은닉 상태를 얼마나 '무시'할지를 정합니다. 만약 r이 0이면, 식 (3)으로부터 , 새로운 은닉 상태는 입력만으로 결정됩니다.

update 게이트는 은닉 상태를 갱신하는 게이트입니다. LSTM의 forget게이트와 input게이트라는 두 가지 역할을 혼자 담당합니다. forget 게이트로써의 기능은 식 (4)의 (1-z)부분입니다. 이 계산에 의해 과거의 은닉 상태에서 잊어야 할 정보를 삭제합니다. 그리고 input 게이트로써의 기능은 뒷부분입니다. 이에 따라 새로 추가된 정보에 input 게이트의 가중치를 부여합니다.

### GRU 코드

```python
class GRU:
    def __init__(self, Wx, Wh):
        '''

        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(3개 분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가중치 매개변수(3개 분의 가중치가 담겨 있음)
        '''
        self.Wx, self.Wh = Wx, Wh
        self.dWx, self.dWh = None, None
        self.cache = None

    def forward(self, x, h_prev):
        H, H3 = self.Wh.shape
        Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H:2 * H], self.Wx[:, 2 * H:]
        Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H:2 * H], self.Wh[:, 2 * H:]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz))
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr))
        h_hat = np.tanh(np.dot(x, Wx) + np.dot(r*h_prev, Wh))
        h_next = (1-z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        H, H3 = self.Wh.shape
        Wxz, Wxr, Wx = self.Wx[:, :H], self.Wx[:, H:2 * H], self.Wx[:, 2 * H:]
        Whz, Whr, Wh = self.Wh[:, :H], self.Wh[:, H:2 * H], self.Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat =dh_next * z
        dh_prev = dh_next * (1-z)

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dWh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1-z)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWx))
        self.dWh = np.hstack((dWhz, dWhr, dWh))

        return dx, dh_prev
```

### LSTM vs GRU

- LSTM은 input, forget, output의 3개의 게이트를 사용하고, GRU는 reset, update의 2개의 게이트를 사용합니다. 게이트의 수가 적은 만큼 학습되는 파라미터 또한 적어서 학습 속도가 더 빠릅니다.
- 데이터 수가 적은 경우 GRU가 대체적으로 성능이 좋습니다. 또한 모델의 구조가 GRU가 더 단순하므로 구조 변형이 더 용이합니다.
- 긴 시퀀스에 대한 학습에서는 LSTM이 대체적으로 성능이 좋습니다.
- 따라서 주어진 문제와 하이퍼파라미터 설정에 따라 LSTM과 GRU의 성능이 달라집니다.

[When to use GRU over LSTM?](https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

GRU가 처음 제안된 논문입니다. 글을 작성할 때 GRU가 제안된 부분을 참고했습니다.
