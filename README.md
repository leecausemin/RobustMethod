# Robust Deepfake Detection AI

LGrad와 NPR 모델을 사용한 강건한 딥페이크 탐지를 위한 Test-Time Adaptation 방법 (NORM, SGS)

## 개요

이 프로젝트는 다음 방법들을 통해 손상된 이미지를 처리할 수 있는 강건한 딥페이크 탐지 방법을 구현합니다:
- **NORM (Normalization-based Test-Time Adaptation)**: 분포 변화를 처리하기 위해 추론 시 BatchNorm 레이어를 적응시킴
- **SGS (Stochastic Gradient Smoothing)**: Artifact 공간 앙상블을 위해 Anisotropic Huber-TV 디노이징 사용

지원하는 베이스 모델:
- **LGrad** (CVPR 2023): GAN 생성 이미지 탐지를 위한 Gradient 학습
- **NPR** (CVPR 2024): 일반화 가능한 딥페이크 탐지를 위한 이웃 픽셀 관계

## 목차
- [설치](#설치)
- [시작하기](#시작하기)
  - [1. 저장소 클론](#1-저장소-클론)
  - [2. 모델 Git Clone](#2-모델-git-clone)
  - [3. 모델 가중치 다운로드](#3-모델-가중치-다운로드)
  - [4. 데이터셋 다운로드](#4-데이터셋-다운로드)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [참고자료](#참고자료)

---

## 설치

### 요구사항
- Python 3.7+
- PyTorch 1.8+
- CUDA (GPU 지원용)

### 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/leecausemin/RobustMethod.git
cd RobustMethod
```

### 2. 모델 Git Clone

LGrad와 NPR 원본 저장소를 클론해야 합니다:

```bash
# LGrad 클론
cd model/LGrad
git clone https://github.com/chuangchuangtan/LGrad.git lgrad
cd ../..

# NPR 클론
cd model/NPR
git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git npr
cd ../..
```

### 3. 모델 가중치 다운로드

#### LGrad 가중치

**방법 1: 수동 다운로드**
1. [Google Drive](https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix?usp=share_link)에서 사전학습된 가중치 다운로드
2. 압축 해제 후 `model/LGrad/weights/` 에 배치

**방법 2: 명령줄 (gdown 사용)**
```bash
# gdown 설치
pip install gdown>=4.7.1

# LGrad 가중치 다운로드
cd model/LGrad/weights
gdown --folder https://drive.google.com/drive/folders/17-MAyCpMqyn4b_DFP2LekrmIgRovwoix
cd ../../..
```

**StyleGAN Discriminator 가중치 (LGrad에 필수)**
```bash
cd model/LGrad/weights
wget https://lid-1302259812.cos.ap-nanjing.myqcloud.com/tmp/karras2019stylegan-bedrooms-256x256.pkl
cd ../../..
```

#### NPR 가중치

NPR 가중치는 이미 저장소에 포함되어 있습니다:
- `model/NPR/weights/NPR.pth`
- `model/NPR/weights/model_epoch_last_3090.pth`

가중치가 없다면 [NPR GitHub](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)에서 다운로드하여 `model/NPR/weights/` 에 배치

### 4. 데이터셋 다운로드

**CNNDetection 데이터셋 다운로드 및 Corruption 자동 적용**

```bash
bash run_download_and_corrupt.sh
```

이 스크립트는 자동으로:
1. Hugging Face에서 CNNDetection 데이터셋 다운로드
2. `dataset/test/` 에 압축 해제
3. 6가지 corruption 타입 적용 (contrast, gaussian_noise, motion_blur, pixelate, jpeg_compression, fog)
4. `corrupted_data_<gan_type>/` 에 손상된 이미지 저장

#### 다운로드 후 데이터셋 구조

```
dataset/
└── test/
    ├── progan/
    ├── stylegan/
    ├── stylegan2/
    ├── biggan/
    └── ...

corrupted_data_<gan_type>/
├── original/           # 원본 이미지
├── contrast/          # 대비 조정
├── gaussian_noise/    # 가우시안 노이즈
├── motion_blur/       # 모션 블러
├── pixelate/          # 픽셀화
├── jpeg_compression/  # JPEG 압축
└── fog/              # 안개 효과
```

---

## 사용법

### 예제 노트북 실행

예제 노트북이 제공됩니다:
- `example.ipynb` - 기본 사용 예제
- `example_SGS.ipynb` - SGS 방법 예제
- `example_SGS_NORM.ipynb` - SGS와 NORM 결합 예제

```bash
jupyter notebook example.ipynb
```

---

## 프로젝트 구조

```
RobustMethod/
├── model/
│   ├── LGrad/
│   │   ├── lgrad_model.py          # LGrad 모델 래퍼
│   │   ├── weights/                 # LGrad 모델 가중치
│   │   │   ├── LGrad-Pretrained-Model/
│   │   │   └── karras2019stylegan-bedrooms-256x256_discriminator.pth
│   │   └── lgrad/                   # 원본 LGrad 구현
│   ├── NPR/
│   │   ├── npr_model.py             # NPR 모델 래퍼
│   │   ├── weights/                 # NPR 모델 가중치
│   │   │   ├── NPR.pth
│   │   │   └── model_epoch_last_3090.pth
│   │   └── npr/                     # 원본 NPR 구현
│   └── method/
│       ├── norm.py                  # NORM (Test-Time Adaptation)
│       └── sgs.py                   # SGS (Stochastic Gradient Smoothing)
├── data_download/
│   ├── apply_corruptions.py         # 이미지 corruption 함수
│   ├── process_dataset.py           # 데이터셋 처리 스크립트
│   └── README.md
├── download_cnndetection_and_corrupt.py  # 자동 다운로드 스크립트
├── run_download_and_corrupt.sh      # 데이터셋 다운로드 실행 스크립트
├── example.ipynb                    # 기본 사용 예제
├── example_SGS.ipynb                # SGS 예제
├── example_SGS_NORM.ipynb           # 결합 방법
├── test_unified_methods.py          # 유닛 테스트
└── README.md                        # 이 파일
```

---

## Corruption 타입

5가지 강도 레벨 (1=약함, 5=강함)로 6가지 corruption 타입 지원:

1. **Contrast**: 대비 조정
2. **Gaussian Noise**: 가산 가우시안 노이즈
3. **Motion Blur**: 방향성 블러
4. **Pixelate**: 다운샘플링 및 업샘플링
5. **JPEG Compression**: JPEG 압축 아티팩트
6. **Fog**: 대기 안개 효과

---

## 방법론

### NORM (Normalization-based Test-Time Adaptation)

분포 변화를 처리하기 위해 테스트 시 BatchNorm 레이어를 적응:
- source와 target의 가중 조합으로 running statistics 업데이트
- corruption으로 인한 분포 변화에 특히 효과적
- LGrad와 NPR 모두 지원

### SGS (Stochastic Gradient Smoothing)

Artifact 앙상블을 위한 엣지 보존 디노이징:
- K개의 다른 Anisotropic Huber-TV 디노이징 적용
- 고분산 노이즈를 줄이기 위해 artifact 평균화
- corruption 노이즈를 제거하면서 진짜 artifact 신호 보존

---

## 참고자료

### 논문

**LGrad** (CVPR 2023)
```bibtex
@inproceedings{tan2023learning,
  title={Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection},
  author={Tan, Chuangchuang and Zhao, Yao and Wei, Shikui and Gu, Guanghua and Wei, Yunchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12105--12114},
  year={2023}
}
```

**NPR** (CVPR 2024)
```bibtex
@inproceedings{tan2024rethinking,
  title={Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection},
  author={Tan, Chuangchuang and Liu, Huan and Zhao, Yao and Wei, Shikui and Gu, Guanghua and Liu, Ping and Wei, Yunchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

### 원본 저장소

- [LGrad GitHub](https://github.com/chuangchuangtan/LGrad)
- [NPR GitHub](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [CNNDetection GitHub](https://github.com/peterwang512/CNNDetection)

### 데이터셋

- [CNNDetection on Hugging Face](https://huggingface.co/datasets/sywang/CNNDetection)
- [NPR 테스트 데이터셋](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)

---

## 문제 해결

### 일반적인 문제

**1. CUDA 메모리 부족**
- 예측 시 배치 크기 줄이기
- CPU 모드 사용: `device="cpu"`

**2. 가중치 누락**
- `model/LGrad/weights/` 와 `model/NPR/weights/` 디렉토리 확인
- 위의 Google Drive 링크에서 재다운로드

**3. 데이터셋 다운로드 실패**
- 인터넷 연결 확인
- Hugging Face에서 수동 다운로드 시도

**4. Import 오류**
- 프로젝트 루트 디렉토리에 있는지 확인
- 모든 요구사항 설치: `pip install -r requirements.txt`

---

## 라이선스

이 프로젝트는 다음을 기반으로 합니다:
- [LGrad](https://github.com/chuangchuangtan/LGrad) - 원본 라이선스 적용
- [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) - 원본 라이선스 적용

이 코드를 사용하는 경우 원본 논문을 인용해주세요.

---

## 연락처

질문이나 이슈가 있으면 [GitHub 저장소](https://github.com/leecausemin/RobustMethod)에 이슈를 열어주세요.
