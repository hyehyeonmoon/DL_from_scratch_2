# Ch04 word2vec 속도 개선

생성일: 2021년 2월 1일 오후 4:41
태그: 재이 김

## [4장 word2vec 속도 개선]

3장에서 구현한 CBOW 모델은 어휘 수가 많아지면 계산량도 커진다는 문제점이 있습니다. 따라서 4장에서는 word2vec의 속도 개선을 목표로 하고 두 가지 개선을 추가했습니다. 

입력층에서 단어를 원핫 표현으로 다루기 때문에 어휘 수가 많아지면 벡터의 크기도 커집니다. 행렬 곱 계산이 아닌, 단어 ID에 해당하는 행(벡터)을 추출하는 Embedding 계층을 만들면 이 문제가 해결됩니다.

Negative Sampling의 핵심은 '다중 분류'문제를 '이진 분류'방식으로 해결하는 것입니다. 이를 손실함수로 사용하면 어휘가 아무리 많아져도 계산량을 낮은 수준에서 일정하게 억제할 수 있습니다. 또, 모든 부정적 예를 대상으로 학습하는 방식은 비효율적이기 때문에 네거티브 샘플링은 소수의 부정적 예시로 학습합니다.

자연어 처리 분야에서 단어의 분산 표현이 중요한 이유는 전이학습(transfer learning)에 있습니다. 단어의 분산 표현은 다양한 자연어 처리 작업에 이용할 수 있습니다.

## [정리]

- Embedding 계층은 단어의 분산 표현을 담고 있으며, 순전파 시 지정한 단어 ID의 벡터를 추출한다.
- word2vec은 어휘 수의 증가에 비례하여 계산량도 증가하므로, 근사치로 계산하는 빠른 기법을 사용하면 좋다.
- 네거티브 샘플링은 부정적 예를 몇 개 샘플링하는 기법으로, 이를 이용하면 다중 분류를 이진 분류처럼 취급할 수 있다.
- word2vec으로 얻은 단어의 분산 표현에는 단어의 의미가 녹아들어 있으며, 비슷한 맥락에서 사용되는 단어는 단어 벡터 공간에서 가까이 위치한다.
- word2vec의 단어의 분산 표현을 이용하면 유추 문제를 벡터의 덧셈과 뺄셈으로 풀 수 있게 된다.
- word2vec은 전이 학습 측면에서 특히 중요하며, 그 단어의 분산 표현은 다양한 자연어 처리 작업에 이용할 수 있다.

## [파일 설명]

- negative_sampling_layer : Embedding Dot 계층, UnigramSampler 클래스, 네거티브 샘플링 클래스가 구현된 파일입니다.
- cbow : SimpleCBOW 클래스를 개선한 모델을 구현한 파일입니다.
- train : CBOW 모델을 학습하는 파일입니다.
- cbow_params.pkl : train 파일을 학습완료한 매개변수를 저장한 파일입니다.
- eval : 학습한 단어의 분산 표현을 평가하는 파일입니다.
- skip_gram : simple_skip_gram을 개선한 파일입니다.

## [심화]

### 임베딩의 종류와 성능

**1) 행렬 분해**

- Corpus 정보가 들어 있는 원래 행렬을 Decomposition을 통해 임베딩하는 기법이다. Decomposition 이후엔 둘 중 하나의 행렬만 사용하거나 둘을 sum하거나 concatenate하는 방식으로 임베딩을 한다.
- Ex) GloVe, Swivel 등

**2) 예측 기반**

- 어떤 단어 주변에 특정 단어가 나타날지 예측하거나, 이전 단어들이 주어졌을 때 다음 단어가 무엇일지 예측하거나, 문장 내 일부 단어를 지우고 해당 단어가 무엇일지 맞추는 과정에서 학습하는 방법
- Ex) Word2Vec, FastText, BERT, ELMo, GPT 등

**3) 토픽 기반**

- 주어진 문서에 잠재된 주제를 추론하는 방식으로 임베딩을 수행하는 기법
- Ex) LDA

### LDA란?

### **1) LDA (Latent Dirichlet Allocation, 잠재 디리클레 할당)**

- 토픽 모델링은 문서의 집합에서 토픽을 찾아내는 프로세스를 말합니다. 이는 검색 엔진, 고객 민원 시스템 등과 같이 문서의 주제를 알아낼 때 사용됩니다.
- LDA는 문서들은 토픽들의 혼합으로 구성되어져 있으며, 토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정합니다. 데이터가 주어지면, LDA는 문서가 생성되던 과정을 역추적합니다

### **2) LDA의 개요**

문서1 : 저는 사과랑 바나나를 먹어요

문서2 : 우리는 귀여운 강아지가 좋아요

문서3 : 저의 깜찍하고 귀여운 강아지가 바나나를 먹어요

- 먼저 사용자가 문서집합에서 몇 개의 토픽을 찾을 지 설정합니다. 예시에서는 2개를 찾겠습니다.
- 문서에서 불필요한 조사 등을 제거한 전처리 과정을 거쳤다고 가정하고 이를 입력으로 넣으면 LDA는 각 문서의 토픽 분포와, 각 토픽 내의 단어 분포를 추정합니다.

**<각 문서의 토픽 분포>**

문서1 : 토픽 A 100%

문서2 : 토픽 B 100%

문서3 : 토픽 B 60%, 토픽 A 40%

**<각 토픽 내의 단어 분포>**

토픽A : 사과 20%, 바나나 40%, 먹어요 40%, 귀여운 0%, 강아지 0%, 깜찍하고 0%, 좋아요 0%

토픽B : 사과 0%, 바나나 0%, 먹어요 0%, 귀여운 33%, 강아지 33%, 깜찍하고 16%, 좋아요 16%

- LDA는 토픽의 제목을 정해주지는 않지만, 사용자는 위의 결과로부터 두 토픽이 각각 '과일'과 '강아지'에 대한 토픽이라고 판단할 수 있습니다.

### **3) LDA 수행하기**

**1. 사용자는 알고리즘에게 토픽의 개수 k를 알려줍니다.**

- 앞서 말하였듯이 LDA에게 토픽의 개수를 알려주는 역할은 사용자의 역할입니다. LDA는 토픽의 개수 k를 입력받으면, k개의 토픽이 M개의 전체 문서에 걸쳐 분포되어 있다고 가정합니다.

**2. 모든 단어를 k개 중 하나의 토픽에 할당합니다.**

- 이제 LDA는 모든 문서의 모든 단어에 대해서 k개 중 하나의 토픽을 랜덤으로 할당합니다. 이 작업이 끝나면 각 문서는 토픽을 가지며, 토픽은 단어 분포를 가지는 상태입니다. 물론 랜덤으로 할당하였기 때문에 사실 이 결과는 전부 틀린 상태입니다. 만약 한 단어가 한 문서에서 2회 이상 등장하였다면, 각 단어는 서로 다른 토픽에 할당되었을 수도 있습니다.

![initial](https://user-images.githubusercontent.com/66687384/106584031-11e3d180-6589-11eb-8e96-07453aec8928.png)

**3. 이제 모든 문서의 모든 단어에 대해서 아래의 사항을 반복 진행합니다**

- 어떤 문서의 각 단어 w는 자신은 잘못된 토픽에 할당되어져 있지만, 다른 단어들은 전부 올바른 토픽에 할당되어져 있는 상태라고 가정합니다. 이에 따라 단어 w는 아래의 두 가지 기준에 따라서 토픽이 재할당됩니다.
- (1) p(topic t | document d) : 문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율
- 우선 첫번째로 사용하는 기준은 문서 doc1의 단어들이 어떤 토픽에 해당하는지를 봅니다. doc1의 모든 단어들은 토픽 A와 토픽 B에 50 대 50의 비율로 할당되어져 있으므로, 이 기준에 따르면 단어 apple은 토픽 A 또는 토픽 B 둘 중 어디에도 속할 가능성이 있습니다.

![initial](https://user-images.githubusercontent.com/66687384/106584036-127c6800-6589-11eb-8b6c-9394b338f7ac.png)

- (2) p(word w | topic t) : 단어 w를 갖고 있는 모든 문서들 중 토픽 t가 할당된 비율
- 두번째 기준은 단어 apple이 전체 문서에서 어떤 토픽에 할당되어져 있는지를 봅니다. 이 기준에 따르면 단어 apple은 토픽 B에 할당될 가능성이 높습니다. 이러한 두 가지 기준을 참고하여 LDA는 doc1의 apple을 어떤 토픽에 할당할지 결정합니다.

![initial](https://user-images.githubusercontent.com/66687384/106584040-1314fe80-6589-11eb-8443-502406813c47.png)

- 이 과정을 반복하면, 각 문서는 모든 할당이 완료된 수렴 상태가 됩니다.

### 참고 사이트 및 이미지 출처

[임베딩의 종류](https://heung-bae-lee.github.io/2020/01/16/NLP_01/)

[LDA](https://wikidocs.net/30708)
