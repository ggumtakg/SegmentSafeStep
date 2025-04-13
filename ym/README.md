- Rule base 보행가능구역 레이블링
  - 파일명 : Depth_Anything_Labeling.ipynb
  - 인도 전체 + 차도 외곽을 커널로 보행가능구역 지정
  - depth anything v2 모델을 활용해 depth 정보로 커널의 크기 조절
  - 자동차, 사람 등의 움직일 수 있는 장애물 주변 구역지정해제
 
- 시연용 프로그램
  - 파일명 : segstep.py
  - PyQt를 사용하여 세그먼테이션 모델을 카메라를 통해 실시간 인퍼런스하는 앱

  - 파일명 : pidstep.py
  - 모델을 bisenet에서 pidnet으로 바꾼 프로그램
 
  - 파일명 : bisestep.py
  - bisenetv2+depth_anything_v2_vits 를 통해 깊이비례 보행가능구역 세그먼테이션(1070ti에서 약 13프레임)
 
  - 파일명 : imagestream.py
  - cityscapes시연용 stuttgart 이미지를 영상처럼 재생하는 프로그램(약 30프레임)

비고
- 각 모델 가중치는 아래 깃헙에서 다운받아 사용했습니다.
- [bisenet github](https://github.com/CoinCheung/BiSeNet)
- [pidnet github](https://github.com/XuJiacong/PIDNet)
- [depth_anything_v2](https://github.com/DepthAnything/Depth-Anything-V2)
