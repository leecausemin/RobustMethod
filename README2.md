# Frequency-Domain Gradient Normalization (FreqNorm)

## 핵심 아이디어 (한 줄 요약)

**Corruption은 LGrad의 gradient map이 가지는 주파수 에너지 분포를 왜곡한다.**
**우리는 학습 없이(test-time 연산만으로) gradient의 주파수 에너지를 '클린 기준 분포'로 되돌려, 분류기가 원래 학습된 artifact 패턴을 다시 보게 만든다.**

---

## 문제 인식 (Why)

### LGrad의 동작 원리
1. LGrad는 **gradient map = artifact map**을 분류함
2. Classifier(ResNet50)는 **특정 주파수 구조의 gradient**에 익숙함
3. 학습 데이터(ProGAN clean images)의 gradient 주파수 분포를 학습

### Corruption의 영향
- **Blur / JPEG / Noise**는 gradient의 **band-wise energy 비율**을 깨뜨림
- Fake artifact와 corruption artifact가 섞여 보임
- 예시:
  - JPEG: 고주파 성분 손실 → gradient의 high-frequency energy 감소
  - Gaussian Blur: 고주파 감쇠 → edge gradient 약화
  - Gaussian Noise: 고주파 증폭 → 무의미한 high-frequency 노이즈

### 기존 NORM의 한계
- BatchNorm adaptation은 **전체 스케일/분포만 맞출 뿐**
- **주파수 대역별 왜곡은 복원하지 못함**
- 왜 작동하는지 해석이 어려움 (간접적 보정)

---

## 제안 방법: FreqNorm (What)

### Reference-Matching Band Gain (학습 불필요)

#### 1단계: Reference 통계 수집 (사전 계산)
클린 데이터에서 gradient FFT의 **band-wise energy 비율**을 통계로 저장:
```
E_ref[b] = mean energy of frequency band b (from clean gradients)
```

#### 2단계: Test-Time 보정
입력 이미지의 gradient `g`에 대해:

1. **FFT 변환**: `G = FFT2D(g)`
2. **Band-wise energy 계산**: 각 주파수 대역 `b`에 대해 현재 에너지 `E[b]` 계산
3. **Gain 계산** (닫힌 형태):
   ```
   α_b = clip((E_ref[b] / (E[b] + ε))^ρ, α_min, α_max)
   ```
   - `ρ`: 보정 강도 (0.5~1.0)
   - `α_min, α_max`: over-correction 방지 (0.5~2.0)
4. **주파수 대역별 gain 적용**: `G[b] *= α_b`
5. **IFFT 역변환**: `g_corrected = IFFT2D(G)`

#### 3단계: 기존 파이프라인
```
Image → img2grad → FreqNorm → classifier(+NORM optional)
```

---

## NORM vs FreqNorm 비교

| 항목 | NORM (기존) | FreqNorm (제안) |
|------|-------------|-----------------|
| **타겟** | BatchNorm 통계 (mean, var) | 주파수 대역별 에너지 |
| **작동 위치** | Classifier 내부 (BN layers) | Gradient map 직접 보정 |
| **보정 방식** | 간접적 (BN을 통해 feature 분포 조정) | 직접적 (gradient 주파수 복원) |
| **해석 가능성** | 낮음 ("왜 되는지 불명확") | 높음 ("corruption의 주파수 왜곡 타겟팅") |
| **계산 비용** | 낮음 (forward pass만) | 중간 (FFT/IFFT 추가) |
| **하이퍼파라미터** | source_sum | ρ, α_min, α_max, band 정의 |
| **병행 가능성** | - | FreqNorm + NORM 조합 가능 |

---

## 강점 분석

### 1. 문제 정의가 더 정확함
```python
# NORM: 전체 채널의 mean/var만 조정
running_mean = (1-α) * source_mean + α * batch_mean

# FreqNorm: 주파수 대역별로 에너지 비율 복원
α[b] = (E_ref[b] / E[b])^ρ  # band-wise
```

**Corruption의 본질**을 더 잘 포착:
- JPEG: 고주파 성분 손실
- Gaussian blur: 고주파 감쇠
- Noise: 고주파 증폭

### 2. 해석 가능성
- NORM: "왜 BN 통계를 바꾸면 되는가?" → 간접적, 설명 어려움
- FreqNorm: "corruption이 왜곡한 주파수를 복원" → 직관적

### 3. LGrad와의 연결성
LGrad는 이미 gradient map을 사용 → **주파수 정보가 핵심**임을 시사

---

## 구현 계획

### Phase 1: 가설 검증 ✅ 우선
**목표**: Corruption이 실제로 주파수를 왜곡하는지 검증

#### 실험 1: 주파수 스펙트럼 시각화
```python
# visualize_frequency_analysis.py
for corruption in ["original", "blur", "jpeg", "noise", ...]:
    grad = lgrad.img2grad(img)
    fft = torch.fft.fft2(grad)
    plot_spectrum(fft, title=corruption)
```

**검증 항목**:
- [ ] Corruption별로 FFT spectrum이 다른가?
- [ ] Band-wise energy 비율이 corruption에 따라 변하는가?
- [ ] JPEG/blur는 고주파 감소, noise는 고주파 증가하는가?

---

### Phase 2: 알고리즘 구현

#### Step 1: 주파수 대역 정의
```python
# Logarithmic band split (제안)
FREQUENCY_BANDS = [
    (0, 4),      # DC + very low
    (4, 16),     # low
    (16, 64),    # mid
    (64, 128),   # high
    (128, 256),  # very high
]
```

#### Step 2: Reference 통계 수집
```python
# collect_reference_stats.py
clean_dataset = CorruptedDataset(corruptions=["original"])
E_ref = compute_band_statistics(clean_dataset, lgrad)
torch.save(E_ref, "freq_reference.pth")
```

**질문**: 어떤 데이터를 reference로 사용할 것인가?
- Option 1: ProGAN 학습 데이터 (LGrad가 학습된 데이터)
- Option 2: 각 테스트 데이터셋의 "original" corruption
- Option 3: 혼합 (여러 clean 데이터의 평균)

#### Step 3: FreqNorm 모듈 구현
```python
class LGradFreqNorm(LGrad):
    def __init__(self, E_ref, rho=0.5, alpha_min=0.5, alpha_max=2.0):
        super().__init__()
        self.E_ref = E_ref
        self.rho = rho
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def img2grad(self, x):
        grad = super().img2grad(x)
        grad = self.apply_frequency_gain(grad)
        return grad

    def apply_frequency_gain(self, grad):
        # FFT
        grad_fft = torch.fft.fft2(grad)

        # Band-wise gain
        for band_idx, (low, high) in enumerate(FREQUENCY_BANDS):
            mask = create_band_mask(grad.shape, low, high)
            E_current = (grad_fft[mask].abs() ** 2).mean()

            # Compute gain
            alpha = torch.clamp(
                (self.E_ref[band_idx] / (E_current + 1e-8)) ** self.rho,
                self.alpha_min, self.alpha_max
            )

            # Apply gain
            grad_fft[mask] *= alpha

        # IFFT
        return torch.fft.ifft2(grad_fft).real
```

---

### Phase 3: 비교 실험

#### 실험 설계
1. **Baseline**: LGrad (원본)
2. **NORM**: LGrad + BatchNorm adaptation
3. **FreqNorm**: LGrad + Frequency normalization
4. **FreqNorm+NORM**: LGrad + FreqNorm + BN adaptation (둘 다)

#### 평가 데이터
- 13개 데이터셋 × 7개 corruption = 91개 조합
- 메트릭: Accuracy, AUC, AP, F1

#### 예상 결과
```
                    Original  Blur   JPEG   Noise  Avg
LGrad (baseline)    95%       60%    65%    55%    69%
NORM                95%       75%    78%    70%    80%
FreqNorm            95%       80%    85%    75%    84%  ← 예상
FreqNorm+NORM       95%       82%    87%    78%    86%  ← 예상 최고
```

---

## 잠재적 문제점 및 해결책

### 1. 2D FFT 계산 비용
**문제**: 256×256 이미지에서 batch마다 FFT/IFFT
**해결책**:
- GPU 최적화 (torch.fft는 CUDA 가속 지원)
- Batch 단위 처리
- 필요시 DCT 같은 빠른 대안

### 2. Spatial 정보 손실 가능성
**문제**: 주파수 도메인 gain은 global operation
**해결책**:
- Windowed FFT 또는 Wavelet transform
- 국소적 corruption에 대응

### 3. Reference 선택의 영향
**문제**: 어떤 데이터를 "클린"으로 볼 것인가?
**해결책**:
- Ablation study로 reference 선택 비교
- 각 데이터셋마다 adaptive reference

### 4. 하이퍼파라미터 민감도
**문제**: ρ, α_min, α_max 튜닝 필요
**해결책**:
- Grid search (ρ ∈ [0.3, 0.5, 0.7, 1.0])
- Cross-validation

---

## 논문 기여도

### NORM의 한계
- "왜 되는지 모르겠지만 됨" (empirical)
- BatchNorm이라는 간접적 메커니즘

### FreqNorm의 강점
- **명확한 가설**: "Corruption은 주파수 왜곡"
- **직접적 타겟팅**: Gradient 주파수를 직접 복원
- **해석 가능성**: 주파수 분석으로 설명 가능
- **확장 가능성**: 다른 gradient-based 방법에도 적용 가능

---

## 다음 단계

### Immediate (지금 당장)
- [ ] **Phase 1 검증 코드 작성 및 실행**
  - `visualize_frequency_analysis.py`
  - Corruption별 주파수 스펙트럼 분석
  - Band-wise energy 계산 및 비교

### Short-term (1주 이내)
- [ ] FreqNorm 모듈 구현
- [ ] Reference 통계 수집
- [ ] 단일 corruption에서 proof-of-concept

### Mid-term (1개월 이내)
- [ ] 전체 데이터셋 평가
- [ ] NORM과 성능 비교
- [ ] Ablation study (ρ, band 정의, reference 선택)

### Long-term
- [ ] 논문 작성
- [ ] 다른 deepfake 탐지 모델에 적용
- [ ] Wavelet/DCT 등 다른 주파수 변환 시도

---

## 참고 자료

### 관련 연구
- LGrad: [Learning on Gradients](https://github.com/chuangchuangtan/LGrad)
- Frequency Analysis in Deep Learning: 주파수 도메인에서의 adversarial robustness
- Test-Time Adaptation: TENT, NORM, MEMO 등

### 이론적 배경
- Fourier Transform: 신호의 주파수 분해
- Image Corruptions: ImageNet-C benchmark
- Gradient-based Detection: ProGAN, StyleGAN artifact 분석
