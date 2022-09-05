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



## Self-Attention 동작원리
* 어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구한다. 그리고 구해낸 이 유사도를 가중치로 하여 키와 맵핑되어있는 각각의 '값(Value)'에 반영한다. 그리고 유사도가 반영된 '값(Value)'을 모두 합해 리턴한다.

![](https://wikidocs.net/images/page/22893/%EC%BF%BC%EB%A6%AC.PNG)

* 셀프 어텐션에서는 Q, K, V가 "입력 문장의 단어 벡터들"로 전부 동일한 값을 가진다. 의미적으로 생각해보면, 입력 문장 내의 단어 벡터들끼리 유사도를 구하므로서 단어간의 연관성을 찾아낸다고 볼 수 있다.

#### Q, K, V 벡터 얻기
* 셀프 어텐션은 인코더의 초기 입력인 dmodel의 차원을 가지는 단어 벡터들을 사용해 셀프 어텐션을 수행하는 것이 아니라 우선 각 단어 벡터들로부터 Q벡터, K벡터, V벡터를 얻는 작업을 거친다.  Q벡터, K벡터, V벡터는 기존의 벡터에 dmodel×(dmodel/num_heads) 크기의 각기 다른 3개 가중치 행렬(WQ, WK, WV)을 곱하므로서 구할 수 있다. 이 가중치 행렬은 훈련 과정에서 학습된다. 만약, 논문과 같이 dmodel=512이고 num_heads=8라면, 각 벡터에 3개의 서로 다른 가중치 행렬을 곱하고 64(dK = dmodel/num_heads)의 크기를 가지는 Q, K, V 벡터를 얻는다. 

![](https://wikidocs.net/images/page/31379/transformer11.PNG)

![](https://wikidocs.net/images/page/31379/transformer12.PNG)

#### Scaled dot-product Attention
트랜스포머에서는 어텐션 챕터에 사용했던 내적만을 사용하는 어텐션 함수 score(q,k)=q⋅k가 아니라 여기에 특정값으로 나눠준 어텐션 함수인 score(q,k)=q⋅k/n (아래 예시에서 n = 8)를 사용한다. 이러한 함수를 사용하는 어텐션을 닷-프로덕트 어텐션(dot-product attention)에서 값을 스케일링하는 것을 추가했다고 해 스케일드 닷-프로덕트 어텐션(Scaled dot-product Attention)이라고 한다.

![](https://blogfiles.pstatic.net/MjAyMTExMjJfMTYz/MDAxNjM3NTcwOTA1MTkz.tByP4GK_3OqIMl1u-gmG_QMZgOv_mLh9O_QEZoSCbVwg.8xJ8Tphl7jql-L5tCZNBBIwxBfH8KikX0JWc72N61GUg.PNG.qhruddl51/image.png?type=w1)

* 위와 같은 과정을 모든 단어에 대한 Q벡터에 대해 수행하는 것을 병렬적으로 표현하면 다음과 같다.

![](https://postfiles.pstatic.net/MjAyMTEyMzBfMjIx/MDAxNjQwODMwODMxODgw.gw-Ih_0VHFyJUwJRBY8evfm62hBHqP6uzNnotAc4omog.pDs7yuLJAhQpSWR1Pq86wgvEAdlE8j2aSoWF-5BiNOog.PNG.qhruddl51/image.png?type=w773)

* 결과적으로 나오는 어텐션 값 행렬 a의 크기는 (문장 내 단어의 개수, Q/K/V벡터의 길이) 이다. 


