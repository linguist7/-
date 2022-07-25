# U-Net
* 'U-Net : Convolutional Networks for Biomedical Image Segmentation'은 Image Segmentation를 목적으로 하는 모델의 시초이다.

> ![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2Fafa16f0e-a0da-485d-bf68-8e7f428314d0%2Fimage-segmentation-example-1060x397.jpg.webp)
(출처 : https://viso.ai/deep-learning/image-segmentation-using-deep-learning/)

* Image Segmentation이란 위의 왼쪽 이미지를 받아 오른쪽과 같이 객체의 class별로 구분하는 것이다. 

## introduction
* 이 장에서는 U-Net이 나오게 된 배경을 설명해준다. 
  * 2012년 AlexNet의 등장 이후 컨볼루션 연산을 수행하는 신경망을 여려개 쌓아 이미지를 처리하는 방식을 통해 이미지 속 하나의 이미지가 어떤 객체인지 분류(Classification)하는 과제 큰 성과를 냈다. 그러나 Biomedical image의 경우에는 한 이미지 안에 여러개의 세포가 들어있기 때문에 픽셀별로 클래스 분류를 해야하는, Localization이 포함된 Classification이 필요했고, 이 문제에 대한 해결책으로 픽셀과 픽셀 주변의 영역을 받아 픽셀에 담긴 정보가 어떤 객체를 나타내는건지 판단하는 방식이 제시되었으나, 이렇게 객체를 예측하면 patch별로 연산을 하기 때문에 연산 속도가 매우 느리며 서로 중복된 영역을 가지는 patch가 많아 중복된 예측 결과도 많이 나온다는 단점이 존재했다. 이러한 이유로 인해 나오게 된 U-Net의 특징을 다음과 같이 정리할 수 있다.

* U-Net은 upsampling과정에서 channel의 숫자가 더 많기 때문에 higher resolution layer에 context information을 전파할 수 있다.
* Fully connected layer(FCN layer)를 사용하지 않았기 때문에 patch에서 얻은 정보만 가지고 해당 patch에서 classification을 수행할 수 있고 이 과정을 이미지 전체에 대해 시행하기 때문에 seamless한 segmentation이 가능하다.(GPU 메모리의 한계성 때문에 이 방식 말고는 거대한 이미지를 처리할 수 없다.)
* U-Net은 Data augmentation에 elastic deformation을 적용해 학습용 데이터셋을 많이 늘렸다.
* U-Net은 붙어있는 세포 사이를 구분하는 경계에 높은 가중치를 둔 loss function을 사용했다.

## Network Architecture
* 이 장에서는 U-Net의 구조를 설명해준다.

> ![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2F3dfe09c3-3cd1-44b9-bc2a-680690f18ffa%2F%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-08-31%20%EC%98%A4%ED%9B%84%202.50.03.png)

* U-Net은 사이즈가 줄어드는 경로(contracting path, 왼쪽)와 사이즈가 늘어나는 경로(expansive path, 오른쪽)로 구성되어 있다.

### contracting path
* contracting path는 우리가 아는 전형적인 CNN과 같다. 3x3 사이즈의 kernel을 가진 CNN(3x3 CNN)을 2번 적용하며 이 과정에서 활성화 함수로 ReLU(rectified linear unit)를 사용하고 stride = 2의 2x2 max pooling을 적용해 너비와 높이를 반으로 줄여버린다.(downsampling) 이 과정을 여러번 적용하는데, downsampling을 할 때마다 channel의 크기를 2배로 늘린다.

### expansive path
* expansive path는 contracting path와 좌우로 대칭되는 구조를 가진다. 3x3 CNN을 2번 적용한 뒤 feature map의 너비와 높이를 2배로 늘리는(upsampling) 과정을 여러번 반복하는데 upsampling을 할 때마다 적용하는 CNN에 있는 kerel의 개수를 반으로 줄인다. 이 때, Kernel의 개수를 반으로 줄인 CNN에 적용하기 전에 반대쪽 contracting path에서 같은 층에 있는 feature map과 concatenation한다. 이 과정이 localization에 도움을 준다.

> ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile25.uf.tistory.com%2Fimage%2F996735355DF900C2321D9B)

* 위의 그림을 보면 input size와 output size가 다르다. padding 방식이 다르기 때문이다. 논문에서는 mirror padding 방식을 활용했다. 이미지의 사이즈가 커서 patch단위로 이미지를 인식한다. 노란색 박스가 한번 네트워크를 돌때 인식하는 patch이다. 아웃풋의 공백이 인풋에서는 가장자리로부터 mirror패딩이 되어있다. context를 유지하기 위한 전략이라고 한다.  

## Training
* 이 장에서는 어떻게 학습을 시켰는지 설명해준다.
* GPU 메모리를 최대한 활용하기 위해 이미지 파일을 여러개의 배치로 나눴다. 따라서 학습용 데이터셋의 단위는 이미지가 아닌 이미지 '일부(patch)'인 것이다. 이로 인해 데이터셋 구성에 필요한 이미지 개수는 줄이면서 데이터셋 내 입력 파일의 개수는 늘릴 수 있었다. 그리고 모멘텀을 0.99로 설정해 이전에 조정한 가중치를 현재 시점의 가중치를 조정하는데 상당히 많이 반영했다.
* energy function은 맨 마지막에 얻은 feature map에 픽셀 단위로 soft-max를 수행하고 여기에 cross entropy loss function을 적용한 것이다. 식은 다음과 같다.

> ![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2Fa387d96f-b0da-463e-a8e7-71e3c9f4aa5a%2F%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-09-01%20%EC%98%A4%EC%A0%84%209.28.05.png)

* 위의 식에서 x는 feature map에 있는 각 픽셀을 말한다. 각 픽셀에서 계산한 걸 다 더한 것이다. w(x)는 weight map이라고 픽셀 별로 가중치를 부과하는 역할을 한다. weight map은 같은 클래스 사이에 가질 수 있는 짧은 간격에 대해 학습하게끔 만들어주는 기능하는데, 예를 들어 세포 사이에 떨어진 간격이 짧아 세포별로 구별이 힘든 경우 그 간격에 대해 학습하는 것을 말한다. weight map의 식은 다음과 같다.

> ![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2F360fc1c9-cbfa-4593-b225-7c0c2232ded7%2F%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-09-01%20%EC%98%A4%EC%A0%84%209.30.19.png)

* d1은 바로 가장 가까운 클래스의 테두리와의 거리, d2는 두 번째로 가까운 클래스의 테두리와의 거리를 말한다. 정리하면, energy function은 weight map에 log(픽셀에서 얻은 클래스별 예측값을 soft-max한 것)를 곱한 것이라고 할 수 있겠다.

<hr/>참고자료

### Weight Map

> ![](https://choijhyeok.github.io/paper/images/2022-06-21-U-net/Untitled%204.png)

* a가 원본이미지, b가 원하는 분할 목표, c가 분할된 이미지, d가 wegiht map 시각화 이미지

<hr/>

## Data Augmentation
* Data Augmentation는 학습용 데이터셋이 별로 없을 때 사용하기 유용한 기술이다. 
* U-Net은 Data Augmentation를 위해 shift, rotation, random-elastic deformation이라는 기법을 사용해 Data Augmentation을 구현했다.
  * random-elastic deformation 사용하면 작은 데이터셋에서 segmentation network를 학습할때 좋다함
* Data Augmentation을 구현한 순서는 다음과 같다.
  1. coarse 3 x 3 grid에 random displacement vectors을 이용해 smooth deformation을 수행, displacement는 10개 픽셀이 가지는 값들의 표준편차를 따르는 가우시안 분포에서 임의로 뽑은 값으로 수행한다.
  2. bicubic interpolation을 이용해 픽셀 단위로 displacement를 계산한다.
  3. contracting path의 맨 끝에 있는 Drop out layer가 더욱 implicit한 Data augmentation을 수행한다.

## Experiments
* 이 장은 성능 평가부분이다.

* 우선 U-Net은 전자현미경으로 관찰되는 뉴런 구조에서 cell segmentation task를 수행했고 학습 데이터는 EM segmentation challenge에서 제공된 것이다.
* 평가 지표는 warping error, Rand error, pixel error로 U-Net을 포함한 10개의 모델을 가지고 성능을 평가했고, 평과 결과는 다음과 같다.

> ![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2Ffde09a46-df2a-44ca-9316-a50b753f41dc%2F%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-09-01%20%EC%98%A4%ED%9B%84%208.32.14.png)







