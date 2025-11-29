## 🏗️ SASRecPipeline 사용 설명서

`SASRecPipeline`은 SASRec-Plus 모델을 편리하게 학습·검증·테스트·저장하기 위한 고수준 파이프라인 클래스입니다.  
데이터 전처리만 되어 있다면, 모델 훈련을 한 줄로 실행할 수 있도록 설계되었습니다.

---

### 📌 초기화

```python
pipeline = SASRecPipeline(
    num_items,       # 전체 아이템 개수 (ID는 1~num_items)
    max_len=50,      # 시퀀스 최대 길이 (패딩 포함)
    d_model=64,      # 임베딩 차원
    n_layers=2,      # SASRec encoder layer 개수
    d_ff=None,       # FFN hidden ratio (None이면 d_model*4)
    dropout=0.2,     # Dropout 비율
    share_embd=True, # 임베딩 weight sharing 여부
    fixed_pos_embd=0,# 0: 없음 / 1: sinusoidal add / 2: concat-hybrid
    pad_id=0,        # PAD token index
    lr=1e-3,         # 학습률
    device="cuda",   # 'cpu' or 'cuda'
)
```

### 📂 필드 구성

| 변수명                                     | 설명                                |
| --------------------------------------- | --------------------------------- |
| `model`                                 | SASRec-Plus 모델 객체                 |
| `optimizer`                             | Adam Optimizer                    |
| `train_data` / `val_data` / `test_data` | DataLoader 객체                     |
| `epoch`                                 | 현재까지 학습된 epoch                    |
| `val_ndcg`                              | `{epoch: (hit@k, ndcg@k)}` 형태의 로그 |
| `test_loss`                             | 테스트 loss 저장용 변수                   |


### 🎯 주요 메서드

- `__call__(x)`

	모델 forward 호출을 shortcut으로 지원합니다.

- `model_train(epoch, print_loss=True)`

	지정한 epoch까지 학습합니다.
이미 학습된 epoch가 있다면 이어서 진행됩니다.

- `model_validate(k=10)`

	Validation dataset에 대해 Hit@K / NDCG@K, loss을 계산합니다.

- `model_test(k=10)`

	테스트셋 기준으로 loss, Hit@K 및 NDCG@K 평가합니다.

- `save_model(path)`

	현재 모델 가중치를 저장합니다.

- `load_model(path)`

	저장된 모델 가중치를 복원합니다.


### 🧵 사용 예시

```python
from sasrec_plus import SASRecPipeline

pipeline = SASRecPipeline(num_items=5000, max_len=50)
pipeline.train_data = train_loader
pipeline.val_data = val_loader

pipeline.model_train(epoch=10)
pipeline.model_validate(k=10)
pipeline.model_test(k=10)

```