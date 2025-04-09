
1. 데이터 탐색
    
- AI_HUB 내 활용가능 데이터 세트 탐색
- 인도보행영상, 1인칭 시점 보행영상, 배리어프리존 주행영상이 적합하였으나,
   이면도로를 도보로 걷는 경우로 시나리오를 한정하면서
   인도보행영상 내 서피스 세그멘테이션, 폴리곤세그멘테이션 데이터를 적합한 데이터로 선정  
  
   참조. 서피스세그멘테이션 카테고리별 분포 (alley -> 이면도로)
     
   ![image](https://github.com/user-attachments/assets/18a2cca7-7ca7-4ed2-860e-53b9a4e170d4)

  
   
2. 데이터 라벨링  
- 서피스 세그멘테이션의 22개 클래스를 보행가능/보행불가능 영역인 2개 클래스로 레이블링함(약 18500장)
- 장애물을 detection한 폴리곤 세그멘테이션 이미지중 이면도로가 포함된 이미지 활용
- SAM을 이용한 반자동 라벨링 1000장
- SAM + Point Prompt로 이면도로 마스크 생성
- 라벨링 검수는 직업 수동으로 진행함
- 코드: sam_folder_10point_masked.py (SamPredictor 또는 Lableme 진행은 로컬에서 가능. 코랩에서는 불가)

  ![image](https://github.com/user-attachments/assets/d556c91d-e729-4628-bc2e-ea607ed8c6b4)
   

3. 모델 훈련 및 결과  
- 라벨링된 데이터의 정확도 확인을 위한 모델 훈련 및 결과   
- Pre-trained : Cityscapes 데이터로 훈련된 모델의 가중치 (파일명: model_indobohaeng_why25.pth)
- Fine-tunning: 라벨링된 데이터를 BisenetV2 모델로 훈련함
- Normalization mean = (0.56, 0.56, 0.54), std = (0.20, 0.19, 0.19),
- 저해상도를 고려한 이미지 Resize: 352*352
- patience = 5, threshold = 0.00001  
    
- 코드 : 가중치25_Class2_파인튜닝_250408.ipynb
- Best 모델 가중치: best_model_finetuned_epoch12.pth
- Val Loss: 0.0823, mIoU=0.88, Acc=96.93%, F1=0.9823
  
  ![image](https://github.com/user-attachments/assets/36a02aaa-f61e-4721-9f69-7e457a6cab16)

   ![image](https://github.com/user-attachments/assets/ddf54f4a-adb9-41d8-82e1-409d2536c279)

  

4. 실시간 동영상 테스트  
- 이면도로 보행하는 실시간 동영상 테스트
- 앞서 훈련된 모델의 best 가중치를 통해 실제 동영상에서 보행가능영역을 구분하도록 함

- 코드: green_zone_detail.py (동영상 저장은 코랩에서는 어려움.)
   
     ![image](https://github.com/user-attachments/assets/b2bfb555-57c6-4abe-8674-e18733b6191c)
