# U-Net
'U-Net : Convolutional Networks for Biomedical Image Segmentation'은 Image Segmentation를 목적으로 하는 모델의 시초이다.

![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2Fafa16f0e-a0da-485d-bf68-8e7f428314d0%2Fimage-segmentation-example-1060x397.jpg.webp)
(출처 : https://viso.ai/deep-learning/image-segmentation-using-deep-learning/)
<hr/>

Image Segmentation이란 위의 왼쪽 이미지를 받아 오른쪽과 같이 객체의 class별로 구분하는 것이다. 

## introduction
이 장에서는 U-Net이 나오게 된 배경을 설명해준다. 
2012년 AlexNet의 등장 이후 컨볼루션 연산을 수행하는 신경망을 여려개 쌓아 이미지를 처리하는 방식을 통해 이미지 속 하나의 이미지가 어떤 객체인지 분류(Classification)하는 과제를 해결했다. 그러나 Biomedical image의 경우에는 한 이미지 안에 여러개의 세포가 들어있기 때문에 픽셀별로 클래스 분류를 해야하는, Localization이 포함된 Classification이 필요했고, 이 문제에 대한 해결책으로 픽셀과 픽셀 주변의 영역을 받아 픽셀에 담긴 정보가 어떤 객체를 나타내는건지 판단하는 방식이 제시되었으나, 이렇게 객체를 예측하면 patch별로 연산을 하기 때문에 연산 속도가 매우 느리며 서로 중복된 영역을 가지는 patch가 많아 중복된 예측 결과도 많이 나온다는 단점이 존재했다. 이러한 이유로 인해 나오게 된 U-Net의 특징을 다음과 같이 정리할 수 있다.

* U-Net은 upsampling과정에서 channel의 숫자가 더 많기 때문에 higher resolution layer에 context information을 전파할 수 있다.
* Fully connected layer(FCN layer)를 사용하지 않았기 때문에 patch에서 얻은 정보만 가지고 해당 patch에서 classification을 수행할 수 있고 이 과정을 이미지 전체에 대해 시행하기 때문에 seamless한 segmentation이 가능하다.(GPU 메모리의 한계성 때문에 이 방식 말고는 거대한 이미지를 처리할 수 없다.)
* U-Net은 Data augmentation에 elastic deformation을 적용해 학습용 데이터셋을 많이 늘렸다.
* U-Net은 붙어있는 세포 사이를 구분하는 경계에 높은 가중치를 둔 loss function을 사용했다.

## Network Architecture
이 장에서는 U-Net의 구조를 설명해준다.
![](https://velog.velcdn.com/images%2Fminkyu4506%2Fpost%2F3dfe09c3-3cd1-44b9-bc2a-680690f18ffa%2F%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-08-31%20%EC%98%A4%ED%9B%84%202.50.03.png)

U-Net은 사이즈가 줄어드는 경로(contracting path, 왼쪽)와 사이즈가 늘어나는 경로(expansive path, 오른쪽)로 구성되어 있다.

### contracting path
contracting path는 우리가 아는 전형적인 CNN과 같다.
3x3 사이즈의 kernel을 가진 CNN(3x3 CNN)을 2번 적용하며 이 과정에서 활성화 함수로 ReLU(rectified linear unit)를 사용하고 stride = 2의 2x2 max pooling을 적용해 너비와 높이를 반으로 줄여버린다.(downsampling) 이 과정을 여러번 적용하는데, downsampling을 할 때마다 channel의 크기를 2배로 늘린다.
### expansive path
expansive path는 contracting path와 좌우로 대칭되는 구조를 가진다. 3x3 CNN을 2번 적용한 뒤 feature map의 너비와 높이를 2배로 늘리는(upsampling) 과정을 여러번 반복하는데 upsampling을 할 때마다 적용하는 CNN에 있는 kerel의 개수를 반으로 줄인다. 이 때, Kernel의 개수를 반으로 줄인 CNN에 적용하기 전에 반대쪽 contracting path에서 같은 층에 있는 feature map과 합친다. 이를 논문에서는 'concatenation'라고 표현했다. 위의 U-Net의 구조를 나타낸 그림에서 'copy and crop'이 이에 해당된다. 맨 마지막 단계에서 3x3 CNN을 2번 거치고 upsampling 대신 1x1 CNN을 사용해 channel의 개수를 2개로 줄인다. 이렇게 U-Net은 23개의 CNN 레이어들로 구성되어 있고 segmentation map에 seamless tiling을 하기 위해 max pooling을 할 때 너비(x-size)와 높이(y-size)에 모두 적용하는 것이 중요하다.

## Training
이 장에서는 어떻게 학습을 시켰는지 설명해준다.
