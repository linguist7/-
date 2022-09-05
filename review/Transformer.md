# Attention is all you need
## 등장배경
![](https://user-images.githubusercontent.com/90301603/188367414-0e7ee8e5-b207-4ee0-8750-c338446f36fe.png)
> 출처 https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/lecture_notes/Transformer.pdf
* 기존 seq2seq 모델은 인코더-디코더 구조로 구성, 인코더는 입력 시퀀스를 고정된 크기의 context vecter로 압축하고, 디코더는 이 context vecter를 통해 출력 시퀀스를 만들었다.
* 하지만 인코더가 입력 시퀀스를 고정된 크기의 context vecter로 압축하는 과정에서 입력 시퀀스 정보가 일부 손실된다는 단점이 있었고, 이를 보완하는 것이 어텐션 메커니즘이다.
* 어텐션 메커니즘을 사용하더라도 여전히 RNN 계열 모델들은 다음 셀을 계산하기 위해 이전 셀의 값을 필요로 하기기 때문에 병렬처리가 안되서 속도가 느리다는 단점이 있다.
* 이러한 한계점을 극복하기위해 전적으로 어텐션 메커니즘에 의존해서 어텐션만으로 모델을 구성한 것이 트랜스포머이다.


## 모델 아키텍처
![](https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMTAyMjFfMTI3%2FMDAxNjEzODkwMzA3MDg2.YEgNfak_Px_ghw0ugCiSO5nlXx4pZ-4ay_gUikOta0Ug.9rzErqMq7LjbKOegEmzNgX8txLDSimM9v_ORxRw-akYg.PNG.nsm4421%2Fimage.png&type=sc960_832)

* 트랜스포머도 인코더-디코더의 구조를 가짐
* Attention 과정을 여러 레이어에서 반복

## 인코더(encoder)
![](https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMDExMDJfNTcg%2FMDAxNjA0MjQ1NzI2MDg3.AYn-p4Q5qT5GUM7wKk32_qhmqC851CcjFV7B66R7u3Yg.jh9zGs417GxSkg6SJG2n5iQs68Cz5QGgoxg9kLJxpggg.PNG.ehdrndd%2Fimage.png&type=sc960_832)

* 트랜스포머는 RNN을 필요로 하지 않는다. 따라서 입력 문장이 주어지면 문장의 순서에 대한 정보가 부족하다. 이러한 이유로 트랜스포머에서는 Positional Encoding을 사용하여 별도의 위치에 대한 정보를 더해준다.
* 이후 어텐션(Attention)을 수행해주고 나온값과 Residual Connection을 통해서 나온값을 바로 받아서 정규화(Normalization) 과정을 수행한 뒤 결과를 내보낸다.
* 하나의 인코더 Layer 1에서 결과값을 뽑아낼 수 있다. 이때 각 레이어는 서로 다른 파라미터를 가진다.
> 잔여 학습(Residual Learning)은 어떤 값을 레이어를 거쳐서 반복적으로 단순하게 갱신하는 것이 아니라 특정 레이러를 건너 뛰어서 복사가 된 값을 그대로 넣어주는 기법이다. 기존 정보를 입력받으면서 추가적으로 잔여된 부분만 학습되도록 만들기 때문에 전반적인 학습난이도가 낮고 초기모델 수렴속도가 높고 Global optima를 찾기 좋아짐



## 디코더(decoder)
![](https://postfiles.pstatic.net/MjAyMTA1MTBfMTU2/MDAxNjIwNjMyODc2MjQz.i8exIkxnnsBYMp3RPSUF0ch4nePMGm7GtjNwJ7Qx7i4g.SDJ2yW7ZDip3ds_Q5nyILXlEgNG6qjikIPj0pdIeZS0g.PNG.dh0985/image.png?type=w773)
* 가장 마지막 인코더에서 나오게 된 출력값이 디코더에 들어가게 된다. 디코더 파트에선 매번 출력마다 입력 소스 문장 중 초점을 두어야하는 단어를 찾는다. 디코더도 인코더와 마찬가지로 여러 개의 레이어로 구성되어 있다.
* 디코더 또한 마찬가지로 각각 단어 정보를 받아서 각 단어의 상대적인 위치정보를 알기 위해 Positional Encoding을 추가하고 디코더 layer에는 두 개의 attention이 사용된다.


## attention 종류(Encoder Self-Attention / Masked Decoder Self-Attention / Encoder-Decoder Attention )
* 트랜스포머에는 세가지 어텐션 레이어가 사용된다.

![](https://postfiles.pstatic.net/MjAyMTA1MTBfMSAg/MDAxNjIwNjQ1NTg2NDYw.4tCu4d6uZDom324KDwju8-WBiqfMIhfT_foWyy1i6NYg.LzfCfHTaOO7qIV-4OzAq0VWe709QwV7FnVoAndz6Ejsg.PNG.dh0985/image.png?type=w773)

* Encodoer Self-Attention은 인코더의 각각 입력된 단어가 서로에게 어떤 연관성을 가지는지 계산하는 과정
* Masked Decoder Self-Attention은 디코더의 단어들이 먼저 등장한 단어들에 대해서 연관성을 계산. 다시 말해, 각각의 출력단어가 다른 모든 출력단어를 전부 참고하도록 만들지 않고 앞쪽의 단어만 참고할 수 있도록 만듦
* Encoder-Decoder Attention은 디코더가 인코더의 어텐션 밸류들에 대해 연관성을 계산. 쿼리가 디코더에 있고 각각의 키와 벨류가 인코더에 있는 상황. I like you 라는 입력이 들어왔으면 '나는 너를 좋아해'라는 출력값에서 각각의 단어들이 입력 단어들 중 어떤 정보에 좀 더 많은 가중치를 두는 지 구할 수 있어야 함



## Self-Attention의 이점

