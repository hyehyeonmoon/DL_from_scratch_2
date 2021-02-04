# Ch05_RNN

생성일: 2021년 2월 4일 오전 10:57

## [5장 순환신경망(RNN)]

언어 모델은 단어 나열에 확률을 부여하는 모델로, 평가 방법은 Perplexity가 있다.

RNN은 순환하는 경로이다. 같은 가중치를 사용함으로서 은닉 상태(순서)를 기억해 시계열 데이터, 자연어처리에 주로 사용이 된다.

긴 시계열 데이터를 학습할 때는 데이터를 적당한 길이씩 모으고(이를 "블록"이라 한다.), 블록 단위로 BPTT에 의한 학습을 수행한다(=Truncated BPTT)

RNN 구현을 mini-batch size를 처리할 수 있도록 구현하였고, RNN을 이용한 LM 또한 Embedding, RNN, Affine, Softmax 층을 가진 Language Model을 구현하여 PTB dataset으로 실습을 해보았다.

## [정리]

- RNN은 순환하는 경로가 있고, 이를 통해 내부에 "은닉 상태"를 기억할 수 있다.
- RNN의 순환 경로를 펼침으로써 다수의 RNN 계층이 연결된 신경망으로 해석할 수 있으며, 보통의 오차역전파법으로 학습할 수 있다.(=BPTT)
- 긴 시계열 데이터를 학습할 때는 데이터를 적당한 길이씩 모으고(이를 "블록"이라 한다.), 블록 단위로 BPTT에 의한 학습을 수행한다(=Truncated BPTT)
- Truncated BPTT에서는 역전파의 연결만 끊는다.
- Truncated BPTT에서는 순전파의 연결을 유지하기 위해 데이터를 '순차적'으로 입력해야 한다.
- 언어 모델은 단어 시퀀스를 확률로 해석한다.
- RNN 계층을 이용한 조건부 언어 모델은(이론적으로는) 그때까지 등장한 모든 단어의 정보를 기억할 수 있다.

## [파일 설명]

[제목 없음](https://www.notion.so/35231c9dcedc4334ad89eb095c1698d5)

## [심화 내용]

### Beam search Algorithm

**① Greedy decoding**

각각의 단계에서 가장 확률이 높은 단어를 선택하는 것, 문제는 한 번 결정하고 나면 결정을 번복할 수 없음

따라서 NMT에서 Greedy decoding을 사용했을 때, 문제가 발생함

![Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled.png](Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled.png)

**② Exhaustive Search decoding(완전탐색 알고리즘)**

말 그대로 step t에서 완성문장의 모든 확률을 고려해서 선택하는 것=>계산비용이 매우 큼!

**③ Beam search Decoding(ppt 31p 참고)**

- Beam search Decoding : On each step of decoder, keep track of the k(beam size) most probable partial translations(which we call hypotheses)

[Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/_(cs224n-2020-lecture08-nmt).pdf](Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/_(cs224n-2020-lecture08-nmt).pdf)

![Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled%201.png](Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled%201.png)

- 확률이어서 점수는 모두 음수이지만, 더 높은 점수일수록 더 좋은 문장
- 최적의 방안을 보장하지는 못해도 완전탐색 알고리즘보다는 매우 효율적임

- <END> token 다루는 방법 : 다른 hypotheses는 다른 timestep에서 <END>token을 만들 수 있으므로 when a hypothesis produces <END> that hypothesis is complete. Place it aside and continue exploring other hypotheses.
- 작동을 멈추는 기준 : reach timestep T or at least n completed hypotheses
- 문제점 : 더 긴 hypotheses일수록 더 낮은 점수를 가지게 됨(누적합이므로)

→길이로 정규화를 시켜줌

![Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled%202.png](Ch05_RNN%204444c1d94f884a5e8a5feabcf5b53f0c/Untitled%202.png)

## Standford cs224n : 수업 중 나온 좋은 질문들 정리

1. RNN에서 같은 weight을 주는 것은 설계할 때 내가 주는 가정인가요?

not a assumption, it's more a deliberate decision in the design of an RNN

2. RNN에서 같은 weight을 주는 것은 왜 좋은가요?

fixed-window language model에서 단점으로 언급되었듯이 일부 weight이 특정 단어에만 특화되는 문제가 있었는데 weight을 똑같이 하면 모든 단어에 대해서 학습 가증해서 더 generalization,

즉 test에 어떤 문장이 와도 성능이 좋아질 것이다.

3. 단어들은 다 다른데 weight을 똑같이 주면 다양한 단어의 특성이 반영이 안 되는 것 아닌가요?

weight이 아주 큰 matrix라는 점을 감안할 때 많은 단어의 특성을 store 할 수 있고,

단어의 다양성은 weight matrix가 얼마나 그 정보양을 store 할 수 있느냐가 관건이다.

4. What length is the input during training?

Efficiency concern or data based

5. Does Wh(one of weights) depend on the length you used?

no. the model size doesn't increase for longer input. 길어지더라도 새로운 weight이 더 추가되는 게 아니기 때문이다.

6. How do we choose the dimension of the lowercase Es?

자유선택 or download word2vec의 사이즈에 따라서 선택한다.

7. How you decide to break up your batches affects how you learn?

shuffling it differently each epoch

SGD consider : it should be a good enough approximation that over many steps you will minimize you rloss

8. IS it ever practical to combine RNNs with a list of hand-written rules?

ex)had written rules : don't let you sentence be longer than this many words⇒hacky rules

Beam search가 대표적인 예시로 이후 NMT 에서 배운다.