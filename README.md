# Robust Deepfake AI

딥페이크 탐지를 위한 AI 모델 프로젝트

## 설치 방법

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 외부 저장소 클론

이 프로젝트는 다음 외부 저장소들을 사용합니다:

#### 데이터셋 다운로드 도구
```bash
git clone https://github.com/HanSungwoo/TA_p2_datasetload.git data_download
```

#### LGrad 모델
```bash
git clone https://github.com/chuangchuangtan/LGrad.git model/LGrad/lgrad
```

#### NPR 딥페이크 탐지 모델
```bash
git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git model/NPR/npr
```

## 프로젝트 구조

```
.
├── data_download/          # 데이터셋 다운로드 스크립트 (외부 저장소)
├── dataset/                # 데이터셋 폴더
├── corrupted_dataset/      # 손상된 데이터셋
├── model/
│   ├── LGrad/
│   │   └── lgrad/         # LGrad 모델 (외부 저장소)
│   └── NPR/
│       └── npr/           # NPR 모델 (외부 저장소)
├── utils/                  # 유틸리티 함수들
├── example.ipynb          # 예제 노트북
└── requirements.txt       # 필수 라이브러리 목록
```

## 사용 방법

(여기에 프로젝트 사용 방법을 추가하세요)

## 참고

- LGrad: [https://github.com/chuangchuangtan/LGrad](https://github.com/chuangchuangtan/LGrad)
- NPR-DeepfakeDetection: [https://github.com/chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
