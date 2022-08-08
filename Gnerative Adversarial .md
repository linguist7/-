# Gnerative Adversarial Nets(2014)
* GAN은 생성자(Generator)와 판별자(Discriminator) 두개의 네트워크를 활용한 생성 모델이다. 둘은 '경쟁적 관계'로 하나의 목적 함수(objective Function)를 번갈아가며 최적화한다. 

![](https://postfiles.pstatic.net/MjAyMjAzMDRfMTgw/MDAxNjQ2MzkxNDUwMTcx.fkKprptQHwWXoV1GyBFXz-MUj-yIIWPtHYnJeD3XZlYg.Z2ymxJGPpde6x-pBlJiLGX1wFa-wXYl0KoMerIr9Mr8g.JPEG.dldlsduq94/IMG_C9EAD9164EFD-1.jpeg?type=w773)

* Discriminator는 theta(d)로 편미분한 값만큼 maximize하는 방향으로 gradient ascending하고, 
* Generator는 theta(g)로 편미분한 값만큼 minimize하는 방향으로 gradient descending한다. 
* D가 0.5로 수렴할 때까지 학습!
