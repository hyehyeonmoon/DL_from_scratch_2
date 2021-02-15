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

seqseq와 RNNLM의 차이는 인코더에서 만든 h(은닉상태 벡터)를 입력받는다는 점만 다르다.

### 교사 강요(teaching force)

학습과정에서 이전 시점의 디코더 셀의 예측이 틀렸는데 이를 현재 시점의 디코더 입력으로 사용하면 현재 시점의 디코더 예측이 잘못될 수 있고 이는 디코더 전체의 예측을 어렵게하고 학습시간을 늦춘다. 이를 방지하기 위해 디코더 셀의 예측값 대신 실제값을 현재 시점의 디코더 셀의 입력으로 사용하여 학습하는 방식을 **교사강요**방식이라 한다.

![image](https://user-images.githubusercontent.com/63804074/107919023-02fc1680-6fae-11eb-90bf-ca75c104c395.png)
<정확한 예측>
![image](https://user-images.githubusercontent.com/63804074/107919031-07283400-6fae-11eb-8418-db7b0b59de29.png)
<틀린 예측>
![image](https://user-images.githubusercontent.com/63804074/107919045-0db6ab80-6fae-11eb-9bf1-17f5dd8f2e78.png)
<교사강요>


![image](https://user-images.githubusercontent.com/63804074/107918690-7a7d7600-6fad-11eb-9b04-a907cfc168fe.png)
<교사강요 구조>

**교사강요**를 사용한다면
학습이 빠르다는 장점이 있다. 하지만 학습과정에 실제값을 넣어서 예측하기 때문에 예측과정에서는 실제값을 기반으로 예측하기때문에 편향문제가 있을 수 있다.

(T. He, J. Zhang, Z. Zhou, and J. Glass. Quantifying Exposure Bias for Neural Language Generation (2019), arXiv.)
위 논문에 따르면 노출 편향 문젝 생각만큼 큰 영향을 미치지 않는다고 한다.

~~~python
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False #random이 teacher_forcing_ratio보다 작으면 True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing di를 index로한 decoder input을 사용함을 볼 수 있다.

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
~~~


**[출처]**
Sooftware 머신러닝: 
https://blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221790750668&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView

 딥 러닝을 이용한 자연어 처리 입문 : https://wikidocs.net/24996
 What is Teacher Forcing for Recurrent Neural Networks? : https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
 자기회귀 속성과 Teacher Forcing 훈련 방법 : https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-9/05-teacher-forcing
 교사강요 코드: https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ko