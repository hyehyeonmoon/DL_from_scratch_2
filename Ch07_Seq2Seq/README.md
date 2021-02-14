# Ch_07 seq2seq

2021/02/15(월) 박준영 정리

**7장 seq2seq**

시계열 데이터를 다른 시계열 데이터로 변환하는 모델인 seq2seq는 Encoder와 Decoder로 이루어져있다.
Encoder는 RNN을 이용해 h라는 고정길이 벡터로 만든다. 그리고 Decoder의 LSTM에 h를 입력하여 하나씩 순차적으로 출력한다.
이때 샘플마다 Data의 길이가 다를 수 있는데, padding을 사용해서 Data의 길이를 균일하게 맞추어준다. 하지만 이 경우 padding을 이용하여 만든 데이터도 seq2seq가 처리하게 된다. 이를 방지하기 위해 Softmax with loss에 마스크 기능을 추가하여 처리한다.

seq2seq의 학습속도를 개선하기 위한 방법으로 Reverse, peeky, attention이 있다. 

**Reverse**의 경우 입력 데이터의 순서를 반전시켜 단어 사이의 평균은 유지시키면서 대응하는 단어와 변환된 단어의 거리를 가깝게 만들어 기울기 전파가 잘 이루어져 학습효율이 좋아지게 만드는 기법입니다.
**Peeky**는 Encoder의 출력 h(중요한 정보를 담고 있는)를 다른 계층에 전달하여 Encoder의 정보를 멀리 있는 decoder에게 잘 전달해준다. 하지만 peeky를 사용하게 되면 가중치 매개변수가 커져서 계산량이 늘어나는 단점이 있다.


**[정리]**

- RNN을 이용한 언어 모델은 새로운 문장을 생성할 수 있다.
- 문장을 생성할 때는 하나의 단어(혹은 문자)를 주고 모델의 출력(확률 분포)에서 샘플링하는 과정을 반복한다.
- RNN을 2개 조합함으로써 시계열 데이터를 다른 시계열 데이터로 변환할 수 있다.
- seq2seq는 Encoder가 출발어 입력문을 인코딩하고, 인코딩된 정보를 Decoder가 받아 디코딩하여 도착어 출력문을 얻는다.
- 입력문을 반전시키는 기법(Reverse), 또는 인코딩된 정보를 Decoder에 여러 계층에 전달하는 기법(Peeky)는 seq2seq의 정확도 향상에 효과적이다.
- 기계번역, 챗봇, 이미지 캡셔닝 등 seq2seq는 다양한 애플리케이션에 이용할 수 있다.




**[파일 설명]**

seq2seq.py : seq2seq 모델구현한 코드입니다.
<br>
generate_better_text/generate_text : BetterRnnlmGen/RnnlmGen으로 text를 생성하는 코드입니다.
<br>
train_seq2seq : seq2seq 모델을 학습한 코드입니다.
<br>
peeky_seq2seq.py: seq2seq peeky를 적용한 코드입니다.

rnnlm_gen.py/rnnlm.py/Rnnlm.pkl/better_rnnlm.py : 위의 코드를 구동시키기위한 py파일입니다.




**[심화]**

### seqseq_Decoder vs RNNLM

![image](https://user-images.githubusercontent.com/63804074/107877700-5b310b00-6f11-11eb-811d-37bb020c0190.png)
<seqseq_decoder>

![image](https://user-images.githubusercontent.com/63804074/107877690-4e141c00-6f11-11eb-83f8-4c66cee5e25d.png)
<RnnLM>

seqseq와 RNNLM의 차이는 인코더에서 만든 h(은닉상태 벡터)를 입력받는다는 점만 다르다. 이 차이는 평범한 언어 보델



**[출처]**
Sooftware 머신러닝: 
https://m.blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221784419691&proxyReferer=https:%2F%2Fwww.google.com%2F
 딥 러닝을 이용한 자연어 처리 입문 : https://wikidocs.net/46496