# Ch08 Attention

생성일: 2021년 2월 19일 오후 4:06
태그: 재이 김

## [8장 Attention]

어텐션은 필요한 정보에만 주목하여 그 정보로부터 시계열 변환을 수행하는 구조를 뜻한다.

seq2seq를 개선하기 위해서, 먼저 Encoder의 출력의 길이를 입력 문장의 길이에 따라 바꿔준다. Decoder에서는 각 단어의 중요도를 나타내는 가중치와 각 단어의 벡터로부터 가중치의 합을 계산해서 맥락 벡터를 구하고, 이 계층을 Attention 계층으로 구현한다.

어텐션과 관련된 기술들로는 양방향 RNN과 Skip 연결 등이 있다.

어텐션은 GNMT, 트랜스포머, NTM 등에 응용된다.

## [정리]

- 번역이나 음성인식 등, 한 시계열 데이터를 다른 시계열 데이터로 변환하는 작업에서는 시계열 데이터 사이의 대응 관계가 존재하는 경우가 많다.
- 어텐션은 두 시계열 데이터 사이의 대응 관계를 데이터로부터 학습한다.
- 어텐션에서는 (하나의 방법으로서) 벡터의 내적을 사용해 벡터 사이의 유사도를 구하고, 그 유사도를 이용한 가중합 벡터가 어텐션의 출력이 된다.
- 어텐션에서 사용하는 연산은 미분 가능하기 때문에 오차역전파법으로 학습할 수 있다.
- 어텐션이 산출하는 가중치(확률)를 시각화하면 입출력의 대응 관계를 볼 수 있다.
- 외부 메모리를 활용한 신경망 확정 연구 예에서는 메모리를 읽고 쓰는 데 어텐션을 사용했다.

## [파일 설명]

- attention_layer : WeightSum, AttentionWeight, Attention, TimeAttention 클래스를 구현한 코드입니다.
- attention_seq2seq : AttentionSeq2Seq 클래스를 구현한 코드입니다.
- train : attention_seq2seq를 학습시킨 코드입니다.
- visualize_attention : Attention의 가중치를 시각화한 코드입니다.

## [심화]

### Self Attention의 의미

앞서 배운 어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구합니다. 그리고 구해낸 이 유사도를 가중치로 하여 키와 맵핑되어있는 각각의 '값(Value)'에 반영해줍니다. 그리고 유사도가 반영된 '값(Value)'을 모두 가중합하여 리턴합니다.

![Untitled](https://user-images.githubusercontent.com/66687384/108496577-9787b100-72ed-11eb-8755-50a845de62c9.png)

 이 때의 Q, K, V를 정의하면 다음과 같습니다.

- Q = Query : t 시점의 디코더 셀에서의 은닉 상태 (찾아야 할 값)
K = Keys : 모든 시점의 인코더 셀의 은닉 상태들 (후보값, Q에 얼마나 연관되어 있는 지)
V = Values : 모든 시점의 인코더 셀의 은닉 상태들

그런데 t 시점이라는 것은 계속 변화하면서 반복적으로 쿼리를 수행하므로 결국 전체 시점에 대해서 일반화를 할 수도 있습니다.

- Q = Querys : 모든 시점의 디코더 셀에서의 은닉 상태들
K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
V = Values : 모든 시점의 인코더 셀의 은닉 상태들

 이처럼 기존에는 디코더 셀의 은닉 상태가 Q이고 인코더 셀의 은닉 상태가 K라는 점에서 Q와 K가 서로 다른 값을 가지고 있었습니다. 그런데 셀프 어텐션은 어텐션을 자기 자신에게 수행하기 때문에 Q, K, V가 전부 동일합니다. 트랜스포머의 셀프 어텐션에서의 Q, K, V는 아래와 같습니다.

- Q : 입력 문장의 모든 단어 벡터들
K : 입력 문장의 모든 단어 벡터들
V : 입력 문장의 모든 단어 벡터들

![Untitled 1](https://user-images.githubusercontent.com/66687384/108496606-a2424600-72ed-11eb-90bf-8cb046c2f78d.png)

셀프 어텐션은 입력 문장의 각 단어 벡터들로부터 Q벡터, K벡터, V벡터를 얻는 작업을 거치고 행렬 연산을 통해 결과값을 냅니다.

![Untitled 2](https://user-images.githubusercontent.com/66687384/108496607-a2dadc80-72ed-11eb-990c-e5fe76c81b8e.png)

예시) I am a student 이라는 문장이 있으면 I , am, a, student들은 각각의 Q, K, V 벡터를 얻게 된다.

![Untitled 3](https://user-images.githubusercontent.com/66687384/108496608-a2dadc80-72ed-11eb-8484-1125eb978679.png)

어텐션 스코어를 이용해 어텐션 분포를 구하고, 이를 사용하여 모든 단어에 대한 어텐션 값을 구한다.

* 어텐션 함수로는 Scaled dot-product Attention을 사용

### 셀프 어텐션의 효과

![Untitled 4](https://user-images.githubusercontent.com/66687384/108496603-a1111900-72ed-11eb-8d04-a6ace4bf19cf.png)

 위의 예시 문장을 번역하면 '그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤하였기 때문이다.' 라는 의미가 됩니다. 여기서 그것(it)에 해당하는 것은 과연 길(street)일까요? 동물(animal)일까요? 사람은 피곤한 주체가 동물이라는 것을 아주 쉽게 알 수 있지만 기계는 그렇지 않습니다. 하지만 셀프 어텐션은 입력 문장 내의 단어들끼리 유사도를 구하므로서 그것(it)이 동물(animal)과 연관되었을 확률이 높다는 것을 찾아냅니다.

### 참고 사이트 및 이미지 출처

[위키독스](https://wikidocs.net/31379)

[NLP Study (5.1)transformer](https://jeonyoonhoi.github.io/study/2019/04/28/study-nlp-transformer/)

[딥러닝 자연어처리 - RNN에서 BERT까지](https://www.slideshare.net/deepseaswjh/rnn-bert)
