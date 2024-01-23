# 문장내 개체간 관계 추출

* 문장내 개체간 관계 추출(Relation Extraction)은 어떠한 문장 내에서 두 단어를 선택했을 때 의미적으로 어떤 관계를 가지는가를 추측하는 Task이다.
* 예를 들어, "브라운은 판교에 산다"는 문장에서 브라운과 판교의 관계는 사람-거주지의 관계가 될 것이다.
* 좀 더 발전하면, 단어간 관계를 모아서 인공지능에게 지식 체계를 만들어 줄 수도 있다. 이것을 지식 그래프(knowledge graph)라고 한다.
![f938e6c9-4e60-4b5f-ac1d-7ac225430acb](https://github.com/boostcampaitech6/level2-klue-nlp-08/assets/22702278/2defa003-a6bd-40cb-9c38-a0fdd3fe0acf)
* 이를 통해 고도화된 인공지능은 위의 질문처럼 어려운 질문에도 대답할 수 있어진다.

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

## 실행방법
- 환경 설치
  
  `pip install -r requirements.txt`
- 학습

  `python train.py`
- 추론

  `python inference.py`
## 프로젝트 구조
```
level2-klue-nlp-08/
│
├── train.py                // 학습 시작을 위한 메인 스크립트
├── train_sweep.py          // train에 하이퍼파라미터 최적화를 더한 코드.
├── train_configs.py        // train에 설정값 받아오는 부분을 더한 코드.
│
├── inference.py            // 학습 모델의 평가 및 추론을 위한 스크립트
├── inference_configs.py    // inference에 설정값 받아오는 부분을 더한 코드.
├── inference_ensemble.py   // inference에 앙상블을 적용해서 여러 모델의 결과값을 조합할 수 있게 된 코드.
│
├── data/                   // 데이터셋 클래스
│   └── dataset.py
├── model/                  // 허깅페이스에서 모델을 불러오는 함수
│   └── model.py
├── preprocessing/          // 데이터셋 전처리, 토크나이징
│   └── preprocessing.py
│   └── tokenizer.py
├── scripts/                // Confusion Matrix와 EDA, 허깅페이스에 모델 업로드하는 코드, 언더샘플링 코드
│   ├── undersampling.py
│   ├── upload_model.py
│   └── classification_evaluation/
│       └── classification_evaluation.py
├── trainer/                // Focal Loss를 적용한 트레이너 클래스
│   └── trainer.py
├── training_recipes/       // 설정값 파일들(.yaml)이 있는 폴더
└── utils/                  // 유틸리티 함수
    ├── utils.py
    └── metrics.py
```
