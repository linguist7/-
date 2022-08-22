# 논문리뷰 - WaveNet: A Generative Model for Raw Audio
* WaveNet은 auto-regressive하게 음성 waveform을 생성하는 dilated causal convolution 기반의 딥러닝 모델입니다.

## sort summary
* WaveNet은 자연스러운 음성 파형을 직접 생성합니다.
* 긴 음성 파형을 학습하고 생성할 수 있는 새로운 구조를 제시합니다.
* 학습된 모델은 컨디션 모델링으로 인해 다양한 특징적인 음성을 생성할 수 있습니다.
* 음악을 포함한 다양한 음성 생성분야에서도 좋은 성능을 보입니다.

# WaveNet
* WaveNet은 음성 파형을 조건부 확률(conditional probability)을 이용하여 다음과 같이 나타냅니다.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">&#x220F;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>T</mi>
    </mrow>
  </munderover>
  <mrow data-mjx-texclass="ORD">
    <mi>p</mi>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>x</mi>
      <mi>t</mi>
    </msub>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <msub>
      <mi>x</mi>
      <mn>1</mn>
    </msub>
    <mo>,</mo>
    <mo>&#x2026;</mo>
    <mo>,</mo>
    <msub>
      <mi>x</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>t</mi>
        <mo>&#x2212;</mo>
        <mn>1</mn>
      </mrow>
    </msub>
    <mo stretchy="false">)</mo>
  </mrow>
</math>

### Dilated Causal Convolutions
* 조건부 확률을 이용하기 위해 Causal 컨볼루션을 사용해 모델이 데이터를 모델링하는 순서를 위반하지 않도록 합니다.
* wavenet은 아래의 그림과 같이 각각의 음성샘플이 이전시간의 샘플들을 조건으로 합니다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcx4XUt%2FbtqFGMPQSAh%2FGjHVj89aEZzLvQAEOWkIr0%2Fimg.png)


*  Causal 컨볼루션이 적용되었기 때문에 반복적인 연결(recurrent connection)을 가지지 않습니다. 하지만 수용 영역(receptive field)을 키우기 위해서는 더 많은 계층이나 더 큰 필터가 필요하다는 것이 causal connection이 가진 문제점입니다. 본 논문에서는 중요도에 따라 수용 영역을 증가시키기 위하여 dilated 컨볼루션을 사용합니다.

![](https://blog.kakaocdn.net/dn/buoEjd/btryd1Td9KY/cIrAcPWs4FDeSqIDRrcosK/img.gif)

* Dilation 계수를 지수적으로(exponentially) 증가시키면 네트워크의 깊이와 수용 영역 역시 지수적으로 커집니다
* Dilation 블록을 쌓는 것은 모델의 능력과 수용 영역의 크기를 더 증가시킬 수 있습니다.

### Softmax Distributions
* 일반적으로 오디오는 16비트 정수값(시간 단계당 1개)의 시퀀스(sequence)로 저장되기 때문에, 소프트맥스 계층은 가능한 모든 확률값을 모델링하기 위해 시간 단계당 56,636개의 확률을 계산해야 합니다. 이를 계산 가능하게 하기 위해 데이터에 μ-law companding transformation을 적용하여 25개의 가능한 값으로 양자화(qunatize)합니다.
> 양자화(quantization) 무한대의 값을 유한한 몇 가지의 대표값으로 바꾸어 주는 것(E.g. 0.5 → 1)

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>x</mi>
    <mi>t</mi>
  </msub>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mtext>sign</mtext>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>x</mi>
    <mi>t</mi>
  </msub>
  <mo stretchy="false">)</mo>
  <mfrac>
    <mrow>
      <mi>ln</mi>
      <mo data-mjx-texclass="NONE">&#x2061;</mo>
      <mrow data-mjx-texclass="ORD">
        <mn>1</mn>
        <mo>+</mo>
        <mi>&#x3BC;</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">|</mo>
        </mrow>
        <msub>
          <mi>x</mi>
          <mi>t</mi>
        </msub>
        <mo stretchy="false">|</mo>
      </mrow>
    </mrow>
    <mrow>
      <mi>ln</mi>
      <mo data-mjx-texclass="NONE">&#x2061;</mo>
      <mo stretchy="false">(</mo>
      <mn>1</mn>
      <mo>+</mo>
      <mi>&#x3BC;</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </mfrac>
</math>

### Gated Activation Units
* wavenet은 gated activation unit을 사용합니다.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">z</mi>
  </mrow>
  <mo>=</mo>
  <mi>tanh</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
  <mo>&#x2299;</mo>
  <mi>&#x3C3;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>g</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
</math>

* 여기서 *는 합성곱(convolution) 연산, ⊙은 원소별 곱(element-wise multiplication) 연산, σ(⋅)은 시그모이드 함수, k는 계층의 인덱스, f는 필터, g는 게이트, W는 학습 가능한 컨볼루션 필터를 의미합니다. 초기 실험에서 이 비선형 함수가 ReLU 함수보다 더 잘 작도하는 것을 확인했습니다.

### Residual and Skip Connections
* wavenet은 수렴 속도를 높이고 더 깊게 모델을 학습하기 위해 residual connection과 매개변수화된 skip connection을 네트워크 전체에 사용하였습니다. 구조는 다음과 같습니다.
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fto7HJ%2Fbtrydzb3Omf%2FFIR5h49YAN3a95Wrluih31%2Fimg.png)

### Conditional WaveNets
* WaveNet은 추가적인 입력값 h가 주어지면, 오디오의 조건부 분포 p(x|h)를 모델링할 수 있습니다. h를 이용하여 앞서 나온 식을 아래와 같이 수정할 수 있습니다.
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">h</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">&#x220F;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>T</mi>
    </mrow>
  </munderover>
  <mrow data-mjx-texclass="ORD">
    <mi>p</mi>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>x</mi>
      <mi>t</mi>
    </msub>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <msub>
      <mi>x</mi>
      <mn>1</mn>
    </msub>
    <mo>,</mo>
    <mo>&#x2026;</mo>
    <mo>,</mo>
    <msub>
      <mi>x</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>t</mi>
        <mo>&#x2212;</mo>
        <mn>1</mn>
      </mrow>
    </msub>
    <mo>,</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">h</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
</math>

* 본 논문에서는 WaveNet에 2가지 다른 방식으로 입력값을 집어넣었습니다. Global conditioning은 모든 시간 단계에 걸쳐 출력 분포에 영향을 미치는 단일 잠재 표현(latent representation) h에 따라 특성을 부여합니다. Global conditioning의 대표적인 예로 TTS 모델에서 발화자 임베딩(speaker embedding)이 있습니다. 앞서 나온 수식에 활성화 함수를 추가하면 다음과 같이 수식을 수정할 수 있습니다.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">z</mi>
  </mrow>
  <mo>=</mo>
  <mi>tanh</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo>+</mo>
    <msubsup>
      <mi>V</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
      <mrow data-mjx-texclass="ORD">
        <mi>T</mi>
      </mrow>
    </msubsup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">h</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
  <mo>&#x2299;</mo>
  <mi>&#x3C3;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>g</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo>+</mo>
    <msubsup>
      <mi>V</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
      <mrow data-mjx-texclass="ORD">
        <mi>T</mi>
      </mrow>
    </msubsup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">h</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
</math>

* local conditioning의 경우, 오디오 신호보다 낮은 샘플링 주파수(sampling frequency)의 두 번째 시계열(timeseries) ht를 가질 수 있습니다. 먼저 오디오 신호와 동일한 해상도로 새로운 시계열 y=f(h)에 맵핑하는 transposed 컨볼루션 네트워크를 사용하여 두 번째 시계열을 변환합니다. 이후, 다음의 식과 같이 활성화 단위에서 사용합니다.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">z</mi>
  </mrow>
  <mo>=</mo>
  <mi>tanh</mi>
  <mo data-mjx-texclass="NONE">&#x2061;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo>+</mo>
    <msub>
      <mi>V</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>f</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">y</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
  <mo>&#x2299;</mo>
  <mi>&#x3C3;</mi>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">(</mo>
    <msub>
      <mi>W</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>g</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo>+</mo>
    <msub>
      <mi>V</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>g</mi>
        <mo>,</mo>
        <mi>k</mi>
      </mrow>
    </msub>
    <mo>&#x2217;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">y</mi>
    </mrow>
    <mo stretchy="false">)</mo>
  </mrow>
</math>

## Text-to-Speech
*  TTS 문제를 위해 WaveNet은 입력 텍스트로부터 파생된 언어적 특징(linguistic feature)에 대하여 부분적으로 조절합니다. 또한, 언어적 특징에 더하여 로그 단위의 기본 주파수 값(logarithmic fundamental frequency, log⁡F0)으로 조절 가능하도록 학습합니다. 언어적 특징으로부터 log⁡F0 값과 음소 duration을 예측하는 외부의 모델을 학습합니다. TTS 문제를 위해 WaveNet의 성능 평가를 위해 주관적인 비교와 mean opinion score (MOS) 테스트를 진행했습니다. MOS 테스트의 결과는 아래의 표와 같습니다.
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fm3cIb%2FbtryjEQVSQt%2FyFM4GuRkWq9R5V2XSaHPaK%2Fimg.png)
