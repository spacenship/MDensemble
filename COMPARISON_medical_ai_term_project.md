# `protein_dual_flow` vs `Medical-AI-Term-project` 비교 분석

두 프로젝트는 같은 데이터(mdCATH)로 같은 목표(단백질 MD 궤적 생성)를 다루지만,
**무엇을 학습 대상으로 삼는지**가 근본적으로 다르다. 이 문서는 그 차이가
학습 성공/실패, 배치 크기, 입력 설계에 어떻게 이어지는지를 코드와 실측으로 정리한다.

- 비교 대상: `/home/mipstu/wjYang/Medical-AI-Term-project` (이하 **MAI**)
- 우리: `/home/mipstu/wjYang/MolecularDynamics/protein_dual_flow` (이하 **PDF**)
- 작성 시점: 2026-08-05. PDF는 conditioning 붕괴를 수정하고 재학습을 시작한 직후.

---

## 0. 한 장 요약

| 항목 | MAI | PDF |
|---|---|---|
| **FM 출발 분포** | **가우시안 노이즈 → 변위 δ** | 구조 x0 → 구조 x1 |
| 모델이 보는 것 | 깨끗한 구조 + 노이즈 섞인 정답 + 직전 속도 | 보간된 구조 하나 |
| 조건 주입 | `h_i`에 **per-node 덧셈** (embed_dim 전체) | 스칼라 게이트 1개 (수정 전) |
| sinusoidal 주파수 | 1 → 10⁴ **증가**, 입력 정규화 | 1 → 10⁻⁴ **감소**, 원본 입력 (수정 전) |
| 손실 | MSE + **cosine 방향항** | MSE만 (+물리항) |
| 입자 | Cα 1개/잔기 | 백본 4개/잔기 |
| ESM2 | 650M **동결 + 사전계산 캐시** | 150M **in-graph 학습** |
| 은닉 차원 | 256 | 1024 |
| 배칭 | ragged (PyG 방식, 패딩 0%) | dense 패딩 (**낭비 58.9%** 실측) |
| 배치 | 128 (B200 183 GB) | 32 (RTX PRO 6000 95 GB) |
| 프레임 간격 | stride 1 (연속 프레임) | frame_gap 5 |

배치 차이(128 vs 32)는 단일 원인이 아니라 **입자 4배 × 은닉폭 4배 × ESM 학습 여부 ×
패딩 낭비 × VRAM 1.9배**가 누적된 결과다. 학습 성패의 차이는 배치가 아니라
**목적함수 설계와 조건 주입 경로**에서 갈렸다.

---

## 1. 가장 중요한 차이 — flow matching의 출발 분포

### MAI: 노이즈 → 변위 (표준 생성형 FM)

`src/train_decoder_esm2.py:65`

```python
def fm_interpolate(delta, t_scalar, noise_scale=1.0):
    eps = torch.randn_like(delta) * noise_scale
    x_t = (1.0 - t_scalar) * eps + t_scalar * delta
    target = delta - eps
    return x_t, target
```

- `delta = pos_next - pos_t` (Kabsch 정렬·중심화된 변위)
- 출발점은 **가우시안 노이즈**, 도착점은 **변위 그 자체**

그리고 모델 호출(`train_decoder_esm2.py:768`)이 결정적이다:

```python
delta_pos, _ = model(
    f_i, pos_t_c, batch_idx, esm_emb=esm_emb, velocity=velocity,
    noisy_delta=x_t, sigma=t_scalar, temperature=temp,
)
```

모델은 동시에 셋을 받는다:
1. `pos_t_c` — **깨끗한 현재 구조** (보간되지 않은 원본)
2. `x_t` — 노이즈가 섞인 정답 변위
3. `velocity = -(pos_t_c - pos_prev_c)` — **직전 스텝의 변위** (관성 정보)

즉 구조는 항상 정확히 알고, 정답은 `t`에 따라 점진적으로 드러나며, 과거 운동까지 안다.
`t→1`이면 `x_t ≈ δ`라 정답이 거의 노출되고, `t→0`이면 "이 구조에서 평균적으로 어디로
움직이는가"라는 학습 가능한 조건부 기댓값을 배운다.

### PDF: 구조 → 구조

`protein_flow/flow/paths.py:49`

```python
x_tau = (1.0 - tau) * x0 + tau * x1
target_velocity = x1 - x0
```

모델은 **보간된 구조 `x_tau` 하나만** 받는다. `x0`을 따로 보지 못한다.

여기서 문제가 생긴다. `x0`와 `x1`은 같은 단백질의 같은 평형 분포에서 나온 두 형태이고,
그 사이를 선형 보간한 `x_tau` 역시 그럴듯한 형태다. **`x_tau`만 봐서는 어느 방향으로
가는 중인지 알기 어렵다.** 실제로 붕괴한 체크포인트에서 측정한 예측-정답 코사인은

| tau | cosine |
|---|---|
| 0.05 | −0.212 |
| 0.50 | +0.002 |
| 0.95 | +0.213 |

로, |cos| 최대 0.21에 불과했다. 정보가 아예 없지는 않지만 MAI 쪽 설정과 비교하면
과제 자체가 훨씬 어렵다.

> **정리**: MAI는 "노이즈에서 답을 복원하라"는 잘 정의된 생성 문제를 푼다.
> PDF는 "두 평형 상태 사이의 수송을 배워라"는 문제를 푸는데, 두 주변 분포가 사실상
> 동일하므로 참 수송 사상이 항등에 가깝고 학습 신호가 본질적으로 약하다.
> 이것이 PDF에서 `fm_improvement`의 상한이 낮은 근본 이유다.

---

## 2. 조건 주입 — PDF 실패의 직접 대응물

### 2-1. 주파수 배치

MAI (`src/models/decoder.py:929`, `:941`):

```python
freqs = 10.0 ** (torch.arange(half) / max(half - 1, 1) * 4)   # 1 → 10^4 (증가)
args  = t_norm.unsqueeze(-1) * freqs                          # t_norm ∈ [0,1]
```

온도는 `(T - 320) / 130 ∈ [0,1]`로 **정규화 후** 같은 방식으로 임베딩한다.

PDF 수정 전 (`protein_flow/utils.py:58`):

```python
freqs = torch.exp(-math.log(10000) * arange(half) / half)      # 1 → 10^-4 (감소)
args  = x.unsqueeze(-1) * freqs                                # x = tau∈[0,1] 또는 raw Kelvin
```

방향이 정반대다. 감소하는 주파수에 `[0,1]` 입력을 곱하면 거의 모든 대역이
`sin≈0, cos≈1`로 평탄해진다. 실측 비교:

| 방식 | cos(t=0, t=1) | cos(320K, 450K) | 인접 온도(320↔348K) |
|---|---|---|---|
| MAI: `x × (1..10⁴)` | **−0.170** | **−0.170** | **0.101** |
| PDF 수정 전: raw `x × (1..10⁻⁴)` | 0.973 | 0.408 | 0.573 |
| PDF 수정 후: `norm(x)×1000 × (1..10⁻⁴)` | 0.176 | 0.176 | 0.348 |

MAI는 처음부터 올바른 영역에 있었고, PDF의 수정본은 접근은 다르지만(입력을 키움 vs
주파수를 키움) 같은 효과에 도달했다.

한 가지 덧붙이면, MAI의 FM 경로는 실제로는 `time` 임베딩이 아니라 **`sigma` 임베딩**을
쓴다(`use_time_cond`가 아니라 `noisy_delta`+`sigma` 분기로 들어감). 그건
`log(t) / (1..10⁴)`로 감소형이지만, 입력 `log(t)`가 −11.5~0으로 11.5단위를 span하므로
저차 대역이 충분히 분해한다(cos = 0.676). 우리 수정 전 `tau ∈ [0,1]`이 1단위밖에
span하지 못했던 것과 대비된다.

### 2-2. 주입 경로의 폭

MAI (`decoder.py:1014`, `:1029-1039`):

```python
h_i = h_i + geom_feat + sigma_emb        # embed_dim 전체에 per-node 덧셈
...
h_i = h_i + temp_emb[_b]                 # 온도도 per-node 덧셈
```

조건이 **은닉 표현 전체(256차원)에 직접 더해져** 이후 6개 HAMP 층 전부를 통과한다.

PDF 수정 전 (`protein_flow/models/fusion.py`):

```python
gate = torch.sigmoid(self.gate_mlp(...))          # [B, L, 1] — 스칼라 하나
fused = gate * proj_seq(h_seq) + (1-gate) * proj_geo(h_geo)
return self.decoder(h_fused, graph, particle_mask)   # 디코더는 조건을 못 받음
```

입자당 `(0,1)` 스칼라 하나가 두 표현을 섞는 비율만 정한다. **출력의 크기도 부호도
바꿀 수 없다.** 그런데 측정된 최적 배율은 tau에 따라 −86 → +86으로 부호가 뒤집힌다.
표현할 수 없는 것을 요구받으니 낮은 tau와 높은 tau의 그래디언트가 상쇄되고,
예측 크기가 목표의 **0.24%**로 붕괴했다.

PDF는 이번에 FiLM(zero-init)을 추가해 이 자유도를 확보했다.

---

## 3. 손실 함수 — MAI는 붕괴를 명시적으로 방어한다

`src/train_decoder_esm2.py:257`

```python
def delta_loss(delta_pos, target_delta, w_dir=0.2, eps=1e-6):
    """
    2) 방향 손실 — cosine similarity 기반 (델타가 0에 머물 때도 그래디언트 제공)
    """
    lp_pos = nn.functional.mse_loss(delta_pos, target_delta)
    cos = (delta_pos * target_delta).sum(-1) / (target_norm * pred_norm)
    lp_dir = (1.0 - cos).mean()
    return lp_pos + w_dir * lp_dir, lp_pos, lp_dir
```

주석이 그대로 말해준다 — **"델타가 0에 머물 때도 그래디언트 제공"**.
MAI는 zero-collapse를 예상하고 코사인 방향항으로 방어했다. 코사인은 크기에 불변이므로
예측이 0에 가까워도 방향에 대한 그래디언트가 살아 있다.

PDF는 순수 MSE였다. MSE는 목표가 노이즈에 지배될 때 0을 최적해로 만들고, 여기에
`step_scale: 1.0`의 physics 항이 정답 속도에 벌점(tau=0.95에서 bond 1.13, 바닥값의 490배)
까지 얹어 0 쪽으로 더 밀었다. PDF는 이번에 `step_scale_mode: remaining`으로 후자를 고쳤지만,
**코사인 방향항은 아직 없다.** 도입을 검토할 만하다.

기타 손실 비교:

| 항 | MAI | PDF |
|---|---|---|
| 위치/속도 MSE | ✓ | ✓ (`lambda_fm`) |
| 방향 cosine | ✓ `w_dir 0.1~0.2` | ✗ |
| 결합 길이 | ✓ `w_bond 0.1` (Cα-Cα 3.8Å ± 0.3 허용오차) | ✓ `lambda_bond 1.0` (PSF 실제 결합) |
| 결합각 | ✓ `w_angle 0.05` | ✓ `lambda_angle 1.0` |
| 충돌 | ✗ | ✓ `lambda_clash 1.0` |
| ODE 롤아웃 endpoint | ✗ (`rollout_steps 1`) | 있었으나 이번에 제거 |

PDF의 물리항 가중치가 10~20배 크다. 정답에 벌점을 물리던 상황에서는 이 크기가 그대로
붕괴 압력이었다.

---

## 4. 왜 MAI는 배치를 128까지 키울 수 있었나

원인은 하나가 아니라 곱해진다.

### 4-1. 입자 수 — 4배

- MAI: **Cα만** (`src/preprocess/convert_mdcath.py:66`, `positions: (T, N, 3)`에서 N = 잔기 수)
- PDF: **백본 N/CA/C/O**, 잔기당 4개

같은 단백질에서 노드 수가 4배, kNN 엣지 수도 4배다.

### 4-2. 은닉 폭 — 4배

- MAI: `embed_dim 256`, HAMP 6층
- PDF: `hidden_dim 1024`, 서열 4층 + 기하 6층 + 디코더 3층

엣지 MLP 입력은 `2 × hidden`이므로 엣지 활성 메모리는 대략 폭에 비례한다.
실측(512잔기, batch 16): 기하 인코더 층당 **5.5 GiB**, 6층이면 33 GiB로 전체 활성의 60%.

### 4-3. ESM2 — 동결·사전계산 vs in-graph 학습

MAI (`src/models/esm2_encoder.py:70`, `src/preprocess/precompute_esm.py`):

```
ESM-2 650M → 동결(frozen), 출력을 esm_cache_local.pt에 사전 계산
학습 가능 파라미터: esm_proj (Linear 1280→256→embed_dim) ~82k
```

학습 중 ESM은 **호출되지도 않는다.** 캐시에서 `(N, 1280)` 텐서를 읽어 작은 projection만 통과.

PDF: ESM2-150M을 그래프 안에서 forward + backward. 실측 파라미터:

```
total       229.0 M
  ESM2      147.7 M  (전부 학습 대상)
  flow net   81.3 M
```

**229M 전체가 옵티마이저 상태(AdamW 2 모멘트)를 갖는다.** MAI는 사실상 몇 M 수준이다.

### 4-4. 배칭 방식 — 실측 패딩 낭비 58.9%

MAI는 PyG 방식으로 여러 그래프를 하나의 큰 그래프로 이어 붙이고 `batch` 인덱스로 구분한다.
패딩이 **0%**다.

PDF는 `[B, N, 3]` dense 텐서에 마스크를 쓴다. 배치 내 최장 도메인에 맞춰 전부 패딩된다.
실측(batch 32, 15개 배치):

```
real particles      :  238,192
allocated (padded)  :  579,072
padding waste       :  58.9%
per-batch N_max     :  1200 ~ 1208
```

**계산의 59%가 패딩에 쓰인다.** batch 32 중 실질은 약 13개분이다.

### 4-5. 하드웨어 — 1.9배

- MAI: NVIDIA **B200 183 GB**
- PDF: RTX PRO 6000 Blackwell **95 GB**

### 종합

| 요인 | 배율(PDF 불리) |
|---|---|
| 입자/잔기 (1 → 4) | ×4 |
| 은닉 폭 (256 → 1024) | ×4 |
| ESM 학습 여부 | 옵티마이저 상태 229M vs ~수 M |
| 패딩 낭비 | ×2.4 (실효) |
| VRAM | ×0.52 (95/183 GB) |

이걸 곱하면 128 vs 32는 오히려 자연스럽다. PDF는 이번에 기하 인코더 gradient
checkpointing을 넣어 8 → 32로 올렸다(54.8 → 22.8 GiB at batch 16).

---

## 5. 입력 값 비교

| 입력 | MAI | PDF |
|---|---|---|
| 좌표 | Cα, **중심화(centered)** `pos_t_c = pos_t - center` | 백본 4원자, **원본 절대 좌표** (−100~190 Å) |
| 정렬 | Kabsch (`--kabsch_align`) | Kabsch (target에만) |
| 서열 | `f_i (N,34)`: AA one-hot 21 + B-factor + 상대위치 + φ/ψ/ω/τ/θ sin·cos + occupancy<br>(mdCATH 변환 시 이면각은 **0으로 채움**) | `residue_types (B,L)` + ESM 토큰 |
| PLM | `esm_emb (N,1280)` 사전계산 | `esm_input_ids` → in-graph ESM2 → `(B,L,640)` |
| 시간 조건 | `t_scalar ∈ [0,1]` (배치당 스칼라 1개) | `tau ∈ [0,1]` (샘플당) |
| 온도 | `temp` (K) → `(T-320)/130` 정규화 | `temperature` (K) → 수정 전 raw, 수정 후 정규화 |
| 물리 시간 | 없음 (stride 고정) | `physical_delta_t` = 5.0 **상수** (정보량 0) |
| 관성 | `velocity = -(pos_t - pos_prev)` **있음** | **없음** |
| 노이즈 상태 | `noisy_delta = x_t` | 없음 (보간 구조가 대신) |
| 토폴로지 | 없음 (Cα 3.8Å 가정) | PSF 유래 `bond_index`, `angle_index` |

주목할 점 둘:

1. **MAI는 좌표를 중심화한다.** PDF는 −100~190 Å의 원본 박스 좌표를 그대로 넣는다.
   기하 인코더가 상대 벡터만 쓰므로 등변성은 유지되지만, 중심화는 공짜이고 수치적으로
   안전하다. (MAI의 교훈 1번이 "mdCATH CoM drift 최대 102 Å → MSE 11,000 Å²"이다.)
2. **PDF의 `physical_delta_t`는 전 샘플 5.0 상수**다. `frame_gap: 5`, `ps_per_frame: null`이므로
   조건 벡터의 1/3을 정보가 0인 입력이 차지한다.

---

## 6. 데이터 파이프라인

| | MAI | PDF |
|---|---|---|
| 저장 형식 | `.pt` 텐서로 **사전 변환** (`convert_mdcath.py`) | 원본 HDF5 샤드 직접 읽기 |
| 디스크 전략 | 전량 로컬 상주 (Cα만이라 작음) | **청크 로테이션** (다운로드→학습→삭제) |
| 프레임 간격 | `stride 1` — 연속 프레임 | `frame_gap 5` |
| 에폭당 샘플 | `subset_size 50000` 무작위 추출 | 청크 내 궤적당 1쌍 × 20 pass |
| 온도 | 320K + 348K (2종) → 이후 5종 | 320/348/379/413/450K (5종) |
| replica | `n_replicas 5` | 5 |

`stride 1`이 큰 차이다. 연속 프레임 사이 변위는 작고 부드러워 예측 가능성이 훨씬 높다.
PDF의 `frame_gap 5`는 mdCATH 프레임 간격을 고려하면 수 ns에 해당하고, 그만큼 열 요동이
누적되어 본질적으로 예측 불가능한 성분이 커진다.

Cα만 저장하므로 MAI는 3.61 TB를 로테이션할 필요 자체가 없다 — 이것이 PDF의 복잡한
`ShardPool`/manifest 인프라가 MAI에는 없는 이유다.

---

## 7. 공정하게 — MAI도 같은 함정을 겪었다

MAI가 처음부터 잘 된 것은 아니다. `POST_HRISH_EXPERIMENTS.md` 기준:

- **B8–B9 (diffusion)**: σ 조건 스칼라 버그 → `bond_loss 4.0`으로 발산.
  이후 자기회귀 오차 누적으로 **Rg +1699%**.
- **B10**: 자기회귀를 버리고 time-conditional 비자기회귀 `(pos_0, t) → pos_t`로 전환.
  Kabsch 정렬 도입. 여기서 처음으로 안정.
- **B13**: 코사인 LR 스케줄러 wrap-around 버그(`global_step > total_steps`)로 LR이
  2×10⁻⁶에서 시작 → epoch 95에서야 peak. **PDF의 `plateau_patience` 문제와 같은 계열.**
- **FM 단독 변형**(`eval/flowmatch_vs_det_fm_kabsch_2500.json`): `rmsf_r` 0.03~0.22로 부진.
- **`eval/final_fair_comparison.json`**: `diversity_ratio` 0.030~0.119 —
  **참 구조 다양성의 3~12%만 생성**. 즉 MAI도 과소분산(under-dispersion)을 겪었다.

최종적으로 이긴 구성은 순수 FM이 아니라 **FM + time-cond + Collective Motion Module(CMM)
+ 앙상블**이고, 그제서야 `Div`가 1.35(이상적 1.0)에 도달했다.

`sota_eval/final_2tables.md`:

| 데이터셋 | 모델 | RMSF-r ↑ | Div (→1) | JS ↓ | CJ ↑ |
|---|---|---|---|---|---|
| mdCATH val 100 | **Ours-CMM-val200** | **0.7667** | **1.0911** | **0.0665** | **0.8445** |
| mdCATH val 100 | AlphaFlow (ESMFlow-MD) | 0.3041 | 1.6050 | 0.7284 | 0.0873 |
| ATLAS 82 (OOD) | Ours-CMM-val200 | 0.7881 | 1.3523 | 0.0667 | 0.8670 |
| ATLAS 82 (OOD) | AlphaFlow-MD | 0.8141 | 1.3691 | 0.0370 | 0.9081 |

> **즉, "MAI는 잘 되고 PDF는 안 됐다"는 단순한 대비가 아니다.**
> MAI는 17번의 명명된 실험(B1~B17)을 거치며 자기회귀 붕괴, LR 버그, 과소분산을 하나씩
> 제거했다. PDF는 그 여정의 초반 단계에 있고, 이번에 conditioning 붕괴라는 큰 장애물을
> 하나 넘었다.

---

## 8. PDF가 가져올 만한 것

우선순위 순.

### 8-1. 코사인 방향 손실 추가 — 비용 거의 0

MAI의 `w_dir`에 해당하는 항. 크기에 불변이므로 예측이 작아져도 그래디언트가 살아 있다.
zero-collapse에 대한 구조적 보험이며, 이번 수정으로 붕괴가 풀렸어도 재발 방지 가치가 있다.
`flow_matching_loss` 옆에 10줄이면 된다.

### 8-2. `physical_delta_t`를 쓸모 있게 만들거나 빼기

현재 전 샘플 5.0 상수라 조건 벡터의 1/3이 낭비된다. 선택지:
- `mdcath_sampling_max_frame_gap`을 늘려 프레임 간격을 실제로 다양화 → 정보가 생김
- 또는 조건에서 제외

### 8-3. ragged 배칭 검토 — 실효 처리량 2.4배

실측 패딩 낭비 58.9%. dense `[B, N, 3]`을 PyG 스타일 `(sum_N, 3) + batch` 인덱스로 바꾸면
같은 메모리로 실질 잔기 수를 약 2.4배 담을 수 있다. 다만 collate·kNN·Kabsch·물리항이
전부 dense 마스크를 전제하므로 **큰 리팩터링**이다. 지금 런이 끝난 뒤 검토할 항목.

간이 대안: 길이별 버킷 샘플러(비슷한 크기끼리 배치)로 패딩을 크게 줄일 수 있다.
훨씬 적은 변경으로 상당 부분을 회수한다.

### 8-4. 좌표 중심화

`x0`를 배치별 질량중심으로 옮기는 것. 등변성에는 영향 없고 수치적으로 안전하다.

### 8-5. 더 근본적인 질문 — 출발 분포를 바꿀 것인가

PDF의 `x0 → x1` 설정은 참 수송 사상이 항등에 가까워 학습 신호가 본질적으로 약하다.
MAI 방식(노이즈 → δ, 구조를 별도 조건으로 제공)으로 바꾸면 과제가 훨씬 잘 정의된다.
다만 이는 **프로젝트의 정체성 변경**에 가깝다 — PDF의 "두 프레임 사이의 조건부 흐름"이라는
전제 자체를 버리는 것이므로, 현재 런의 결과를 보고 별도로 판단할 사안이다.

중간 지점도 있다: **`x0`를 `x_tau`와 함께 별도 입력으로 넣는 것.** 구조를 바꾸지 않고도
모델이 "어디서 출발했는지"를 알게 되어 `x1 - x0` 추정이 훨씬 쉬워진다. 등변성은
상대 벡터만 쓰면 유지된다. 8-1 다음으로 비용 대비 효과가 클 후보다.

---

## 9. 참고 파일

| 내용 | 경로 |
|---|---|
| FM 보간 | `Medical-AI-Term-project/src/train_decoder_esm2.py:65` |
| FM 학습 분기 | `Medical-AI-Term-project/src/train_decoder_esm2.py:763-780` |
| 방향 손실 | `Medical-AI-Term-project/src/train_decoder_esm2.py:257` |
| 시간/온도 임베딩 | `Medical-AI-Term-project/src/models/decoder.py:929-955` |
| 조건 주입 | `Medical-AI-Term-project/src/models/decoder.py:1008-1039` |
| ESM 동결·캐시 | `Medical-AI-Term-project/src/models/esm2_encoder.py:70`, `src/preprocess/precompute_esm.py` |
| 승리 구성 | `Medical-AI-Term-project/scripts/train_ensemble_fm_collective_full.sh` |
| 실험 연대기 | `Medical-AI-Term-project/POST_HRISH_EXPERIMENTS.md` |
| 최종 성능표 | `Medical-AI-Term-project/sota_eval/final_2tables.md` |
| PDF FM 경로 | `protein_dual_flow/protein_flow/flow/paths.py:49` |
| PDF 조건 주입 | `protein_dual_flow/protein_flow/models/fusion.py` |
| PDF 손실 | `protein_dual_flow/protein_flow/train.py:267` (`compute_losses`) |
