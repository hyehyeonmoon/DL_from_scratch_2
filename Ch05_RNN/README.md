## [5장 순환신경망(RNN)]

- 언어 모델은 단어 나열에 확률을 부여하는 모델로, 평가 방법은 Perplexity가 있다.

- RNN은 순환하는 경로이다. 같은 가중치를 사용함으로서 은닉 상태(순서)를 기억해 시계열 데이터, 자연어처리에 주로 사용이 된다.

- 긴 시계열 데이터를 학습할 때는 데이터를 적당한 길이씩 모으고(이를 "블록"이라 한다.), 블록 단위로 BPTT에 의한 학습을 수행한다(=Truncated BPTT)

- RNN 구현을 mini-batch size를 처리할 수 있도록 구현하였고, RNN을 이용한 LM 또한 Embedding, RNN, Affine, Softmax 층을 가진 Language Model을 구현하여 PTB dataset으로 실습을 해보았다.


## [정리]

- RNN은 순환하는 경로가 있고, 이를 통해 내부에 "은닉 상태"를 기억할 수 있다.
- RNN의 순환 경로를 펼침으로써 다수의 RNN 계층이 연결된 신경망으로 해석할 수 있으며, 보통의 오차역전파법으로 학습할 수 있다.(=BPTT)
- 긴 시계열 데이터를 학습할 때는 데이터를 적당한 길이씩 모으고(이를 "블록"이라 한다.), 블록 단위로 BPTT에 의한 학습을 수행한다(=Truncated BPTT)
- Truncated BPTT에서는 역전파의 연결만 끊는다.
- Truncated BPTT에서는 순전파의 연결을 유지하기 위해 데이터를 '순차적'으로 입력해야 한다.
- 언어 모델은 단어 시퀀스를 확률로 해석한다.
- RNN 계층을 이용한 조건부 언어 모델은(이론적으로는) 그때까지 등장한 모든 단어의 정보를 기억할 수 있다.


## [파일 설명]

|File|Description|
|:-- |:-- |
|simple_rnnlm|rnn language model을 구현한 class|
|train_custom_loop|ptb dataset을 이용해 rnn language model 실습|
|train|trainer 파일을 이용해 체계적인 실습 구현|
|class_summary|Ch05에서 사용한 모든 클래스 및 모델들을 정리해 놓은 파일|

## [심화 내용]

### Beam search Algorithm

**① Greedy decoding**

각각의 단계에서 가장 확률이 높은 단어를 선택하는 것, 문제는 한 번 결정하고 나면 결정을 번복할 수 없음

따라서 NMT에서 Greedy decoding을 사용했을 때, 문제가 발생함

![Untitled](https://user-images.githubusercontent.com/55529617/106932431-1364f380-675b-11eb-9f69-bd68c70bf57f.png)

**② Exhaustive Search decoding(완전탐색 알고리즘)**

말 그대로 step t에서 완성문장의 모든 확률을 고려해서 선택하는 것=>계산비용이 매우 큼!

**③ Beam search Decoding**

- Beam search Decoding : On each step of decoder, keep track of the k(beam size) most probable partial translations(which we call hypotheses)

[_(cs224n-2020-lecture08-nmt).pdf](https://github.com/hyehyeonmoon/DL_from_scratch_2/files/5927508/_.cs224n-2020-lecture08-nmt.pdf)의 beam searching 예시 참고.

![Untitled 1](https://user-images.githubusercontent.com/55529617/106932426-1233c680-675b-11eb-9222-e997d665d277.png)

- 확률이어서 점수는 모두 음수이지만, 더 높은 점수일수록 더 좋은 문장
- 최적의 방안을 보장하지는 못해도 완전탐색 알고리즘보다는 매우 효율적임

- <END> token 다루는 방법 : 다른 hypotheses는 다른 timestep에서 <END>token을 만들 수 있으므로 when a hypothesis produces <END> that hypothesis is complete. Place it aside and continue exploring other hypotheses.
- 작동을 멈추는 기준 : reach timestep T or at least n completed hypotheses
- 문제점 : 더 긴 hypotheses일수록 더 낮은 점수를 가지게 됨(누적합이므로)

→길이로 정규화를 시켜줌

![Untitled 2](https://user-images.githubusercontent.com/55529617/106932428-12cc5d00-675b-11eb-99b9-b9a196aff486.png)


