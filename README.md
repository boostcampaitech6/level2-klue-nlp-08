## 문장 내 개체간 관계 추출 (Relation Extraction)
![f938e6c9-4e60-4b5f-ac1d-7ac225430acb](https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/22702278/2defa003-a6bd-40cb-9c38-a0fdd3fe0acf)

<img width="812" alt="Screenshot 2024-01-24 at 4 11 30 PM" src="https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/76895949/55130a0e-6a1a-457f-a95f-dbcd693eb4d0">

* 문장 내 개체간 관계 추출(Relation Extraction)은 어떠한 문장 내에서 두 단어를 선택했을 때 의미적으로 어떤 관계를 가지는가를 추측하는 Task이다.
* 언어 모델이 주어진 문장을 잘 이해하고 있다면 문장 내 임의의 두 단어들의 관계를 잘 맞출 수 있을 것이다. 이렇게 주어진 문장을 구조화 함으로써 자연어로 표현된 정보 보다 기계가 해석하기 훨씬 수월해 지고 이는 질의응답을 비롯한 다양한 분야에서 유용하게 활용되고 있다.
다양한 방법을 적용해 봄으로써 RE task에서의 모델 성능을 높여본다
* 좀 더 발전하면, 단어간 관계를 모아서 인공지능에게 지식 체계를 만들어 줄 수도 있다. 이것을 지식 그래프(knowledge graph)라고 한다.
* 이를 통해 고도화된 인공지능은 위의 질문처럼 어려운 질문에도 대답할 수 있어진다.

## 프로젝트 내용
* 주어진 dataset에 대해 의미 있는 분석 결과를 도출하여 향후 실험의 방향을 제시한다.
* 필요한 전처리 방법을 도입하고 접목할 수 있는 데이터 증강을 고안한다.
* RE-task에서의 성능 향상을 위해 수행되었던 연구들을 찾아보고 구현하여 실험에 적용한다.
* 언어모델을 전처리한 문장들로 fine-tuning하여 re-task에서의 성능을 향상시킨다.
* hugging face tokenizer, model, trainer 라이브러리에 대해 깊게 이해하고 원하는 목적으로 custom하여 실험을 구축한다.
* 여러 모델들로 앙상블 결과를 도출하여 성능을 올린다.

## 일정

 1월 3일 (수요일) 10:00 \~ 1월 18일 (목요일) 19:00

## 팀원

| 이동근 | 윤석원 | 송영우 | 한혜민 | 김용림 |
| --- | --- | --- | --- | --- |
| <img src="https://avatars.githubusercontent.com/u/22702278?s=64&v=4" width="120" height="120"> | <img src="https://avatars.githubusercontent.com/u/76895949?s=64&v=4" width="120" height="120"> | <img src="https://avatars.githubusercontent.com/u/139039225?s=64&v=4" width="120" height="120"> | <img src="https://avatars.githubusercontent.com/u/105696374?v=4" width="120" height="120"> | <img src="https://avatars.githubusercontent.com/u/125326251?s=64&v=4" width="120" height="120"> |
|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/exena)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/jsdysw)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/ye0ng1)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/hyeming00)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/arles1224)|

## 프로젝트에 사용된 툴
![Sponsorships](https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/22702278/0cf934bf-c26c-4f8e-b6d0-852ed424c138)

## 데이터셋
- Train 데이터 개수: 32470개
- Test 데이터 개수: 7765개
- 한개의 데이터를 이루는 요소: 문장, 두 단어(주체, 객체), Label
- Label: 크게 3개 분류로 나뉘는 총 30개의 관계
  - per: 개인 계열. ex: 사람-생일, 사람-자식, 사람-거주지 등 18개.
  - org: 단체 계열. ex: 회사-창립일, 회사-제품, 사장-직원 등 11개.
  - no_relation: 위에 해당하지 못한 것들.

## 프로젝트 구조
```
level2-klue-nlp-08/
│
├── train.py                // baseline train script
├── train_sweep.py          // sweep을 사용하는 model train script
├── train_configs.py        // config.yaml을 통해 학습을 시작하는 train script
│
├── inference.py            // model inference script
├── inference_configs.py    // config.yaml로 모델을 불러와 inference 수행
├── inference_ensemble.py   // config 파일로 여러 모델을 불러와 앙상블 추론 결과를 얻을 수 있음
│
├── data/                   // 데이터셋 클래스
│   └── dataset.py
├── model/                  // 허깅페이스에서 모델을 불러오는 함수
│   └── model.py
├── preprocessing/          // 데이터 전처리, 토크나이저
│   ├── preprocessing.py
│   └── tokenizer.py
├── scripts/                // 기타 script
│   ├── undersampling.py
│   ├── upload_model.py     // 허깅페이스에 모델 업로드
│   └── classification_evaluation/
│       └── classification_evaluation.py    // 모델 추론 결과를 Confusion Matrix로 출력
├── trainer/                // Focal Loss를 적용한 트레이너 클래스
│   └── trainer.py
├── training_recipes/       // model, train recipe를 config로 담고 있는 파일들
└── utils/                  // 유틸리티 함수
    ├── utils.py
    └── metrics.py
```
## 데이터 분석
<img width="300" alt="스크린샷 2024-01-19 오후 2 30 36 (1)" src="https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/22702278/b46f381a-5eaf-4b15-b9e7-49589cc85470">
<img width="300" alt="스크린샷 2024-01-19 오후 2 39 55" src="https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/22702278/f37035d2-894a-40fe-bd55-2dab79934c47">

라벨은 no_relation이 제일 많았다.
문장 길이는 klue/bert-base의 토크나이저를 통해 토큰화된 단위로 20~60 정도에 대부분 분포되어 있었다.

## 데이터 전처리
- Entity marker 추가
    - Typed Entity Marker Special Token: [S:PERSON] 이순신 [/S:PERSON] 장군은 [O:OCUP] 조선 [/O:OCUP] 출신 이다
    - Typed Entity Marker Punctuation: @ ⊙ 사람 ⊙ 이순신 @ 장군은 # ^ 시대 ^ 조선 # 출신 이다
- 특수문자 제거
    - 영어, 한글, 한자, 일본어와 일부 특수문자를 제거
- train.csv 를 9:1 의 비율로 train 과 valid 데이터셋으로 분리
- 이후 파인튜닝 실험에서 klue-roberta-large에 Special Token을 적용한 것이 F1 스코어 71.01로 제일 좋은 점수가 나왔다.

## 모델 평가 및 개선

- Typed Entity Marking
    - [CLS] 이순신 [SEP] 조선 [SEP] 이순신 장군은 조선 출신이다 [SEP] 라는 문장에 여러 tokenizing 방식을 적용했다. 모델과 실험 환경에 따라 성능 향상폭이 다를 수 있다. 최종 선정 모델들의 경우 punctuation marker가 성능향상을 이끌었다.
    - ConcatEntityTokenizer : [CLS] 이순신 [SEP] 조선 [SEP] 이순신 장군은 조선 출신 이다 [SEP]
    - TypedEntityMarkerTokenizer : [cls] [S:PERSON] 이순신 [/S:PERSON] 장군은 [O:OCUP] 조선 [/O:OCUP] 출신 이다 [sep]
    - TypedEntityMarkerPuncTokenizer : [cls]  ''' @ ⊙ 사람 ⊙ 이순신 @ 장군은 # ^ 시대 ^ 조선 # 출신 이다 ''’ [sep]
    - TypedEntityMarkerTokenizer + one shot : @ ⊙ 사람 ⊙ 이순신 @ 과 # ^ 시대 ^ 조선 # 의 관계는 무엇인가? @ ⊙ 사람 ⊙ 이순신 @ 장군은 # ^ 시대 ^ 조선 # 출신 이다
- Cleaning
    - 정규표현식을 이용해 일부 특수문자, 한글, 숫자, 영어, 중국어, 일본어만 남기는 전처리를 진행했다.
    - vaiv/kobigbird-robert-large을 대상으로 동일한 파라미터에서 raw data와 cleaning한 데이터를 비교했을 때, micro f1 score이 86.674에서 86.52로 미세하게 감소했다.
- Focal Loss
    - 클래스 불균형을 개선하기 위해 Loss Function을 Focal Loss로 적용해봤다. Focal Loss를 통해 모델이 학습할 때 오분류하는 데이터에 집중해서 학습하도록 해준다.
    - vaiv/kobigbird-robert-large을 대상으로 Loss Function을 Cross Entropy와 Focal Loss로 학습한 모델 성능을 비교했다. Cross Entropy보다 Focal Loss로 학습한 모델의 micro f1 score이 86.674에서 86.753으로 소폭 증가했다.
- Ensemble
    - model 과 csv 를 대상으로 hard voting, soft voting 을 수행했다.
- LightGBM
    - lightGBM은 Gradient Boosting 프레임워크로, leaf-wise 방식을 채택한 Tree 기반 학습 알고리즘이다. fine-tuning 이후에 lightGBM으로 모델을 한번 더 학습했다.
- Under sampling
    - no_relation 이 Label인 데이터의 분포가 너무 커서 학습시에 no_relation label 데이터 일부를 drop하여 모델 학습에 활용한다.
- Top k inference
    - no_relation으로 예측한 결과에서 유독 오답이 많았다. 따라서 모델이 no_relation으로 예측 했으나 top-2 예측 확률과 차이가 매우 작은 경우 예측 결과를 교체한다.
