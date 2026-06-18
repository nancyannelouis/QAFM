# CIFAR-100 실험 진행 현황

마지막 업데이트: 2026-06-18 21:31

## 단계별 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| Step 0: 이론 검증 | ✅ 완료 | Theorem 2만 85%(95% 기준 미달) — 측정 방법(커스텀 DCT vs 실제 JPEG 코덱) 한계, 무시 가능 |
| Step 1: 메인실험1 (공격효과성) | ✅ 완료 | **최신 코드로 실행됨 — 재실행 불필요.** clean/qafm/badnets/ftrojan(공식 알고리즘)/blended 전부 정상 |
| Step 2: 메인실험2 (은닉성) | ✅ 완료 | ⚠️ **PSNR 평균 버그가 있던 옛 코드로 실행됨 — 재실행 필요** (BadNets PSNR=inf로 깨짐) |
| Step 3: 메인실험3 (방어저항성) | ✅ 완료 | ⚠️ **Neural Cleanse 샘플 부족 버그가 있던 옛 코드로 실행됨 — 재실행 필요** (128장으로 100클래스 역공학, 클래스당 평균 1.3장) |
| Step 4: Ablation 1 (Component) | 🔄 진행 중 | QAFM 완료(BA=76.86%, ASR@Q50=100%), Fixed-Delta 진행 중. PSNR/NC 버그와 무관한 단계라 재실행 불필요 |
| Step 5: Ablation 2 (k값) | ⏳ 대기 | 시작 전 — 시작 시 자동으로 최신 코드(PSNR 버그 수정 포함) 반영됨 |
| Step 6: Ablation 3 (q_train) | ⏳ 대기 | 시작 전 — 자동으로 최신 코드 반영됨 |
| Step 7: Ablation 4 (poison_rate) | ⏳ 대기 | 시작 전 — 자동으로 최신 코드 반영됨 (0.0=클린 기준선 포함) |

## 다음에 돌릴 때 해야 할 일

### 1. Step 4(ablation, 현재 진행 중)가 끝날 때까지 대기
지금 GPU를 점유하고 있어서 동시에 다른 학습을 돌리면 둘 다 느려짐. Step4~7은 그대로 두면 끝까지 자동 진행됨(`run_all.py`가 한 프로세스로 순서대로 실행 중).

### 2. 전체(혹은 Step4) 끝나면 Step2, Step3만 재실행
```bash
cd cifar100
python experiments/main_exp2_stealth.py --n_samples 1000
python experiments/main_exp3_defense.py --ckpt_dir checkpoints
```
- Step2: BadNets PSNR=inf 버그가 고쳐졌는지 확인
- Step3: Neural Cleanse가 QAFM을 여전히 잡는지(샘플 부족 때문이었는지 진짜 가설 문제인지) 확인. 시간 부담되면 `--nc_samples 500`으로 줄여도 됨

### 3. 위 두 개 다시 돌리고 나면 cifar100 전체 결과가 최신 코드 기준으로 완성됨

## 참고: 왜 Step1은 재실행 안 해도 되는지
FTrojan 공식 알고리즘 포팅, Blended 압축 제거 수정이 **cifar100 실행 시작 전에** 이미 끝나 있었어서, Step1은 처음부터 최신 코드로 돌았음. (cifar10은 반대로 도중에 수정이 들어가서 ftrojan/blended를 따로 백필해야 함 — `cifar10/README.md` 참고)
