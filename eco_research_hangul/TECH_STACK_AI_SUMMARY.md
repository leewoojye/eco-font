# eco_research_hangul 기술스택 및 AI 방법론 요약

작성 기준: 2026-06-26, 현재 `/home/woojye2020/decs_jupyter_lab/eco-font/eco_research_hangul` 구현 기준.

## 핵심 요약

`eco_research_hangul`은 순수한 end-to-end 폰트 생성 모델 하나로 완성 폰트를 만드는 구조가 아니다. 현재 구조는 다음 세 흐름이 같이 들어 있다.

1. 조건부 diffusion 학습 모델: 일반 한글 폰트 글리프를 입력 조건으로 받고, 실제 Nanum Eco 계열 글리프를 목표로 학습한 작은 DDPM.
2. OCR/잉크/스타일 가이드 후보 선택: diffusion 출력, 원본 기반 형태 변환 후보, inline engraving 후보를 만든 뒤 Tesseract OCR, 잉크 절감, Ryman Eco 스타일 점수, 디자인 게이트로 최종 후보를 고르는 reranking 파이프라인.
3. 완전체 TTF 후보 생성: 현대 한글 11,172자 전체를 렌더링하고 erosion, centerline engraving, diagonal perforation 같은 레시피를 적용한 뒤 `fontTools`로 다시 TTF를 만드는 배치 파이프라인.

따라서 현재 방법론의 정확한 성격은 `diffusion-only font generation`보다는 `OCR/ink/style objective 기반 eco glyph optimization + TTF export`에 가깝다.

## 사용 기술스택

| 영역 | 사용 기술 | 역할 |
| --- | --- | --- |
| 언어/실행 | Python 3.10+ | 전체 실험 및 CLI 구현 |
| 딥러닝 | PyTorch, TorchVision | 조건부 DDPM 학습, VGG19 style scorer |
| 생성 모델 | Conditional DDPM | 소스 글리프와 목표 절감률 조건으로 eco glyph 후보 생성 |
| 이미지 처리 | NumPy, OpenCV, scikit-image | erosion, closing, skeletonize, contour, hole/fragment 분석 |
| 글리프 렌더링 | Pillow | TTF/OTF에서 글자 이미지를 96px grayscale bitmap으로 렌더링 |
| 폰트 처리 | fontTools | TTF cmap/glyf 읽기, bitmap contour를 glyph outline으로 변환, 이름 테이블 수정 |
| OCR 평가 | Tesseract OCR | 한글 `kor`, 체르키어 `chr` 인식으로 가독성 게이트/점수 계산 |
| 설정/리포트 | PyYAML, JSONL, tqdm | 실험 설정, manifest, metrics, contact sheet 생성 |

주요 코드 위치:

- 학습 모델: `src/eco_research_hangul/model.py`
- diffusion schedule/sample: `src/eco_research_hangul/diffusion.py`
- 데이터 생성: `src/eco_research_hangul/data_build.py`
- 학습: `src/eco_research_hangul/train.py`
- 기본 추론: `src/eco_research_hangul/infer.py`
- OCR/잉크/스타일 가이드 추론: `src/eco_research_hangul/guided.py`
- 후보 중간결과 생성: `src/eco_research_hangul/candidate_preview.py`
- 지표 계산: `src/eco_research_hangul/metrics.py`
- 완전체 한글 TTF 20개 생성: `scripts/build_full_hangul_ttf_candidates.py`

## 인공지능 기술 상세

### 1. 데이터 구성

학습 데이터는 글자 단위 paired glyph image이다.

- source: 일반 Nanum 계열 한글 폰트.
- target: 실제 Nanum Eco 계열 한글 폰트.
- glyph image: 기본 96x96 grayscale bitmap.
- 조건 채널: `source glyph`, `target_saving`, `x coordinate map`, `y coordinate map`의 4채널.

즉, 모델은 "이 원본 글리프를 목표 잉크 절감률에 맞는 eco glyph로 바꿔라"라는 조건부 이미지 생성 문제로 학습된다.

### 2. 조건부 DDPM

모델은 작은 U-Net 계열 `ConditionalGlyphDDPM`이다.

- noisy target glyph와 condition tensor를 concat해서 입력한다.
- timestep은 sinusoidal embedding 후 MLP를 거쳐 각 residual block에 주입한다.
- encoder/decoder는 Conv2d downsample, ConvTranspose2d upsample, skip connection 구조다.
- normalization은 GroupNorm, activation은 SiLU를 쓴다.
- diffusion beta schedule은 linear schedule이다.
- 학습 손실은 config에 따라 `x0` 예측 MSE 또는 noise `epsilon` 예측 MSE를 쓴다.

현재 `configs/smoke.yaml` 기준:

- timesteps: 48
- prediction_type: `x0`
- epochs: 30
- batch size: 16
- 학습 출력: `runs/smoke/diffusion_best.pt`
- 마지막 validation MSE: 약 0.0070

주의할 점은, 이 diffusion 모델은 Nanum Eco target을 학습했기 때문에 원시 출력이 작은 구멍/점 패턴에 끌리는 경향이 있다. 그래서 현재 파이프라인에서는 diffusion 결과를 최종 폰트로 바로 쓰지 않고 후보 중 하나로만 사용한다.

### 3. 후보 생성

한 글자마다 여러 후보를 만든다.

- source original
- source erosion 1/2회
- source inline engraving
- source erosion + inline engraving
- diffusion sample
- diffusion sample에 closing, erosion, inline engraving을 추가 적용한 변형

최근 완전체 TTF 생성 스크립트는 diffusion sampling을 쓰지 않고, 20개 고정 style recipe를 전체 11,172자에 적용한다.

주요 style recipe:

- `source_erode1`, `source_erode2`: 글자 외곽을 줄여 잉크 면적 감소.
- `source_inline_w1`, `source_inline_w2`: 글자 내부 중심선을 깎아 Ryman Eco 같은 선형 eco typography 느낌을 부여.
- `source_closed_inline_*`: 작은 구멍을 closing한 뒤 내부 라인을 깎아 OCR 안정성을 높이려는 변형.
- `source_diag_*`: 대각선 패턴으로 ink cut을 만드는 실험적 변형.

### 4. OCR/잉크/스타일 목적함수

`guided.py`의 최종 선택은 단일 손실 하나가 아니라 여러 점수의 조합이다.

잉크 절감률:

```text
ink_area(image) = sum(clip(pixel_value, 0, 1))
ink_saving = 1 - ink_area(generated) / ink_area(source)
```

OCR 점수:

- Tesseract OCR을 사용한다.
- 한글은 `kor`, 체르키어는 `chr`.
- config에서 주로 PSM `[8, 6]`을 시도하고 가장 좋은 결과를 고른다.
- exact match, partial match, confidence를 합쳐 후보 점수에 반영한다.

스타일 점수:

- Ryman Eco OTF를 style reference로 렌더링한다.
- TorchVision VGG19 feature map에서 Gram matrix를 계산한다.
- 후보 이미지와 reference style의 Gram distance를 style loss로 쓴다.
- 이는 Gatys et al. neural style transfer의 Gram-style representation을 glyph 후보 선택용 prior로 가져온 것이다.

디자인 게이트:

- 작은 구멍 수.
- 작은 foreground fragment 수.
- channel area ratio.
- 과도한 잉크 제거 oversave penalty.

현재 guided objective는 OCR 통과와 디자인 게이트를 우선하고, 그 안에서 잉크 절감과 Ryman Eco 스타일을 함께 본다.

### 5. TTF 완전체 생성 방식

`scripts/build_full_hangul_ttf_candidates.py`는 현대 한글 완성형 11,172자를 대상으로 동작한다.

처리 순서:

1. 입력 TTF가 현대 한글 11,172자를 모두 가지고 있는지 `cmap`으로 검증.
2. 각 음절을 Pillow로 96px bitmap 렌더링.
3. style recipe를 bitmap에 적용.
4. OpenCV contour tracing으로 bitmap outline 추출.
5. `fontTools.pens.ttGlyphPen.TTGlyphPen`으로 glyf outline 생성.
6. 원본 TTF의 해당 glyph를 교체.
7. family/postscript name을 새 이름으로 바꾸고 TTF 저장.

최근 실행 결과:

- 출력 경로: `outputs/hangul_full_ttf_20_no_ocr`
- 생성 수: 20개 TTF
- glyph coverage: 각 TTF 현대 한글 11,172자
- OCR 평가: 제외
- 총 소요 시간: 약 297.7초

## 참고 논문 및 레퍼런스와 구현 반영점

### Eco typography / ink saving

1. Ryman Eco
   - 링크: https://www.rymaneco.com/about
   - 반영점: 작은 구멍을 뚫는 Ecofont Sans 방향보다, 여러 가는 선/내부 빈 공간을 이용해 미적이고 지속가능한 글꼴을 만드는 방향을 style prior로 삼았다.
   - 코드 반영: `style_reference_font`, VGG Gram style score, inline engraving 후보.

2. Ecofont legibility and toner consumption studies
   - 링크: https://www.mdpi.com/2413-4155/7/1/29
   - 링크: https://ijiemjournal.uns.ac.rs/index.php/jged/article/download/526/602
   - 반영점: 폰트 품질을 SSIM 같은 원본 유사도만으로 보지 않고, 잉크/토너 사용량과 가독성을 같이 평가한다.
   - 코드 반영: `ink_saving`, Tesseract OCR match/confidence, hole/fragment gate.

### Diffusion font generation

1. Ho et al., "Denoising Diffusion Probabilistic Models"
   - 링크: https://arxiv.org/abs/2006.11239
   - 반영점: forward noise process와 reverse denoising process를 학습하는 DDPM 기본 구조.
   - 코드 반영: `DiffusionSchedule`, `q_sample`, `p_sample`, `sample_loop`.

2. Diff-Font: Diffusion Model for Robust One-Shot Font Generation
   - 링크: https://arxiv.org/abs/2212.05895
   - 반영점: 복잡한 문자 체계의 글꼴 생성을 diffusion 조건부 생성 문제로 보는 관점.
   - 차이점: 이 폴더 구현은 Diff-Font의 stroke-wise dataset, content/style/stroke attribute encoder를 그대로 구현하지 않았다. 작은 DDPM 후보 생성기로 단순화되어 있다.

3. FontDiffuser
   - 링크: https://arxiv.org/abs/2312.12142
   - 반영점: font imitation을 denoising diffusion image-to-image generation으로 보는 관점, 복잡한 획 보존의 중요성.
   - 차이점: Multi-scale Content Aggregation, Style Contrastive Refinement는 현재 구현되어 있지 않다.

### Few-shot / localized style font generation

1. LF-Font
   - 링크: https://arxiv.org/abs/2009.11042
   - 링크: https://github.com/clovaai/lffont
   - 반영점: 한 글꼴의 전역 스타일 하나만 보지 않고, 복잡한 문자 구조에서는 local/component-wise style이 중요하다는 관점.
   - 차이점: 현재 구현은 LF-Font의 component factorization 네트워크를 직접 구현하지 않았다. 대신 한글 글리프 내부 중심선, fragment, hole, channel 같은 이미지 구조 지표를 사용한다.

2. clovaai fewshot-font-generation
   - 링크: https://github.com/clovaai/fewshot-font-generation
   - 반영점: FUNIT, DM-Font, LF-Font, MX-Font 등 few-shot font generation 계열을 비교 기준으로 삼을 수 있다.
   - 차이점: 이 폴더는 공식 clovaai 모델을 학습/재현한 것은 아니다.

### Style loss / OCR / image processing

1. Gatys et al., "A Neural Algorithm of Artistic Style"
   - 링크: https://arxiv.org/abs/1508.06576
   - 반영점: VGG feature의 Gram matrix로 visual style을 수치화하는 아이디어.
   - 코드 반영: `VGGStyleScorer`.

2. Tesseract OCR
   - 링크: https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
   - 반영점: OCR 언어팩과 PSM을 바꿔 글리프 가독성을 자동 평가.
   - 코드 반영: `metrics.py`의 `recognize_tesseract`, `recognize_tesseract_multi`.

3. OpenCV morphological operations
   - 링크: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
   - 반영점: erosion, closing, contour, connected component 기반 후보 생성 및 디자인 게이트.

4. fontTools
   - 링크: https://fonttools.readthedocs.io/en/latest/ttLib/ttFont.html
   - 반영점: TTF의 `cmap`, `glyf`, `name` table 처리와 generated glyph 저장.

## 현재 구현의 한계

- 완전체 한글 TTF 20개 생성은 diffusion sampling이 아니라 source bitmap recipe 기반이다.
- OCR 평가는 완전체 11,172자 전체에 대해 수행한 것이 아니라, 별도 selected subset/sample 기반으로 수행했다.
- carbon saving은 별도 물리 모델이나 LCA 측정값이 아니라 ink saving과 동일 비율로 둔 근사치다.
- Ryman Eco style은 VGG Gram score로 약하게 유도될 뿐, Ryman Eco 같은 전문 type design grammar를 직접 학습한 것은 아니다.
- LF-Font, Diff-Font, FontDiffuser 논문 구조를 그대로 재현한 코드는 아니다. 현재 구현은 이 논문들의 관점을 참고한 실험용 하이브리드 파이프라인이다.

## 실험 해석

현재 폴더의 생성모델을 한 문장으로 정리하면 다음과 같다.

```text
원본 한글 TTF를 글리프 bitmap으로 렌더링한 뒤,
diffusion 후보와 형태학적 eco 후보를 만들고,
OCR 가독성 + 잉크 절감 + Ryman Eco 스타일 prior + 디자인 게이트로 후보를 선택하거나,
완전체 11,172자에 고정 style recipe를 적용해 TTF로 재구성하는 실험 파이프라인.
```

연구 방향으로는 "원본 보존"보다 "OCR로 읽히는 범위에서 잉크 절감과 미적 eco typography를 최대화"하는 목표에 맞춰져 있다.
