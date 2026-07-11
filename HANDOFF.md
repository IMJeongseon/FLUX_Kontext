# HANDOFF — 다른 서버에서 이어서 실행할 작업

> 작성: 2026-07-11. 로컬(MLLAB) GPU가 점유 중이라 대기 상태인 실험들.
> 모든 명령은 저장소 루트에서 실행. 결과 판정 기준까지 이 문서에 있음.

## 0. 환경

- **GPU**: 2×24GB 이상 (FLUX.1-Kontext bf16, `device_map=balanced`).
  VRAM이 더 크면 `--max-memory "0:40GiB,1:40GiB"`처럼 조정.
- **Python env** (로컬 `flowedit` env 기준):
  `torch 2.8.0+cu128, diffusers 0.35.1, transformers 4.54.0, scipy 1.17.1,
  torchvision, Pillow, numpy`
- **가중치** (HF gated — 토큰 필요):
  `black-forest-labs/FLUX.1-Kontext-dev` (실험 본체),
  `black-forest-labs/FLUX.1-dev` (T2I ceiling test),
  DINOv2는 torch.hub에서 자동 다운로드.
- **준비 산출물**: `data/handoff/pug_s123/` — Stage-1 결과가 이미 들어 있어
  아래 1번을 바로 실행 가능 (draft.png = 단일 pass 출력 seed123,
  composite.png / omega_mask.png = 관측된 워프 합성).

```bash
# manifest가 참조하는 경로로 draft 복사 (지표 비교용)
mkdir -p outputs/bench outputs/twostage/pug_s123
cp data/handoff/pug_s123/draft.png outputs/bench/pug_jump_ours_s123.png
cp data/handoff/pug_s123/{composite.png,omega_mask.png} outputs/twostage/pug_s123/
```

## 1. [최우선] 해법-A Stage 2: 투영 조화 (pug)

**목적**: "관측된 대응 + latent 투영 = 미세 identity의 *해결*" 검증.
단일 pass(top-k transport)는 점프 성공 시 marker가 미세하게 붕괴한다
(pug "K" — marker_dino 0.80(정지) → 0.62(점프)). Stage 2는 marker 표면을
등식 제약으로 앵커하므로 이 붕괴가 구성적으로 사라져야 한다.

```bash
python src/twostage.py \
  --source data/sources/pug_shirt.png \
  --composite outputs/twostage/pug_s123/composite.png \
  --omega-mask outputs/twostage/pug_s123/omega_mask.png \
  --prompt "The pug wearing a green outfit with the white letter K is jumping high in the air to catch a tennis ball." \
  --output outputs/twostage/pug_s123/final.png
```

**판정**: `final.png`에서 (a) 점프 포즈 유지, (b) "K"가 소스와 동일한
획 구조로 선명, (c) 이음새·조명 자연스러움. 실패 모드별 노브:
경계 티가 나면 `--anchor-core-until 24`, 포즈가 무너지면
`--anchor-all-until 10`.

## 2. OT vs top-k (E_assign 비중 판정)

**목적**: 오차 분해에서 다대일 붕괴 항(E_assign)의 실제 비중 측정.
top-k(열 제약 없음) → OT(Sinkhorn, 행+열 제약)로 바꿨을 때 marker 지표가
오르면 E_assign이 주범, 안 오르면 배치 항(E_arrange)이 지배.

```bash
python src/pipeline.py \
  --input data/sources/pug_shirt.png --mask data/masks/mask_pug_shirt_rough.png --object pug \
  --prompt "The pug wearing a green outfit with the white letter K is jumping high in the air to catch a tennis ball." \
  --seed 123 --identity-ot \
  --output outputs/bench/pug_jump_ours_s123_ot.png

python src/pipeline.py \
  --input data/sources/kitten_sweater.png --mask data/masks/mask_kitten_sweater_rough.png --object kitten \
  --prompt "The kitten wearing a red and white patterned outfit is leaping high in the air to catch a feather toy." \
  --identity-ot \
  --output outputs/bench/kitten_leap_ours_ot.png

python src/pipeline.py \
  --input data/sources/dalmatian_collar.png --mask data/masks/mask_dalmatian_collar_rough.png --object dalmatian \
  --prompt "The dalmatian wearing a yellow collar is jumping high in the air to catch a red frisbee." \
  --identity-ot \
  --output outputs/bench/dalmatian_collar_ours_ot.png
```

## 3. 지표 산출 (1·2 완료 후)

```bash
python eval/metrics.py --manifest eval/solutionA_manifest.json \
  --out outputs/solutionA_metrics.json
```

**기준치 (top-k, 이미 측정됨)**:

| case | marker_dino | marker_color | 비고 |
|---|---|---|---|
| pug/s123_topk | 0.621 | 0.490 | 점프 성공, K 뒤틀림 |
| kitten/topk | 0.798 | 0.512 | 도약 성공, 니트 패턴 유지 |
| dalmatian/topk | 0.877 | 0.701 | 최고 성적 (대조군 — OT가 해치지 않는지 확인) |

**성공 기준**: `pug/s123_projA`(해법 A)가 marker_dino ≥ 0.75 및
marker_color ≥ 0.65로 복원 + motion > 0 유지. OT는 pug/kitten 상승 여부와
dalmatian 무회귀(≥0.85) 확인.

## 4. [후순위] T2I prior ceiling test

**목적**: goat rearing / penguin belly-slide의 motion 실패가 우리 방법
결함인지 base prior 천장인지 판정. 순수 T2I(FLUX.1-dev)로도 그 장면을
못 그리면 → 천장 물려받음(envelope에 명시), 그리면 → 우리 결함.

```bash
python scripts/t2i_ceiling_test.py   # 1 GPU, cpu-offload, ~5분
```

## 5. [백로그, 순서대로]

1. Stage 2가 통과하면: 나머지 실패 케이스에 해법-A 확장
   (kitten/dalmatian은 Stage 1부터 — `src/observe_warp.py --focus-box`는
   `eval/bench_manifest.json`의 marker_box 값 사용)
2. 성공 케이스 포함 전체 10케이스 × 3 seed (42/123/777) 완주 → 성공률 표
3. baseline 3종(plain Kontext `--baseline` / KV-Edit / FlowEdit)을 동일
   10케이스·동일 중립어 프롬프트로 실행 → 비교표
   (KV-Edit: `~/MLLAB/KV-Edit/run_kv_ours.py`, 512² 표준, FLUX_DEV/AE env 필요;
   FlowEdit: `~/MLLAB/FlowEdit/`, 전용 conda env `baselines`)
4. metrics v3: 3×3 grid-cell 색 히스토그램 (배치 민감 지표 — E_arrange 측정)

## 현재까지의 핵심 판정 기록 (요약)

- 벤치마크 10케이스: 성공 5 / motion 미달 3 / identity 실패 2 →
  seed 스윕·프롬프트 수정 후 **8/10** (penguin·goat-rearing만 잔존)
- verb boost(`--motion-boost`): β 2→4 포화 — prior에 없는 layout은
  증폭으로 못 만듦 (커밋 2f1138d)
- **미세 identity 손실의 수학적 진단**: E_mix + E_assign + E_arrange +
  E_quant 분해. bias-only 클래스는 볼록결합+비제어 잔차 경로 때문에
  원리적으로 E > 0 → "해결"은 (대응 관측 + latent 투영) 클래스 전환 필요
  (커밋 a62cf92 = 해법 A 구현)
