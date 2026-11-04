# Quantus 0.6.0과 EXP-01 구현의 대조

**검증 결과:** EXP-01의 NumPy evaluator가 기록한 중간값과 점수를 실제 Quantus 호출과 비교했다. 지정한 fixture의 **34개 점검이 모두 통과**했고, 선택한 8개 설치 소스 파일은 아래 고정 commit의 Git blob과 일치했다. 이는 제한된 입력·설정에서의 구현 대조 결과이며, 개발 실험이나 confirmation 결과를 대신하지 않는다.

EXP-01의 기본 실행 경로는 `exp01-numpy-pearson-v1`이다. Quantus를 필수 의존성으로 두지 않고, 별도 audit에서 공식 구현과 대조한다. 이렇게 하면 각 subset, attribution sum, 실제 모델 출력 변화를 직접 저장하면서도 라이브러리와의 차이를 확인할 수 있다.

## 1. 무엇을 고정했나

확인한 upstream 버전은 **Quantus 0.6.0**, commit [`85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`](https://github.com/understandable-machine-intelligence-lab/Quantus/tree/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a)이다. 최신 버전의 동작에 관한 주장이 아니다.

| 항목 | Quantus FC 기본값 | EXP-01에서 명시한 값과 이유 |
|---|---|---|
| `abs` | `False` | `False`: 기여의 부호를 보존한다. |
| `normalise` | `True` | `False`: 원래 scalar output 단위의 기여를 유지한다. |
| `perturb_baseline` | `"black"` | `0.0`: 설명 reference와 교체 값을 정확히 맞춘다. |
| `return_aggregate` | `True` | `False`: 입력별 점수와 invalid 여부를 보관한다. |
| `nr_runs` | `100` | 각 설정의 budget `B`. audit fixture는 `25`이다. |
| `subset_size` | `224` | 각 설정의 `k`; 반드시 `1 <= k < d`이다. |
| `similarity_func` | Pearson | Pearson을 유지한다. |
| `softmax` 호출 인자 | `False` | `False`: 고정된 scalar target `g(x)`를 비교한다. |

FC는 각 draw에서 `np.random.choice(..., replace=False)`로 서로 다른 좌표 `k`개를 고른다. 다음 draw에서는 같은 subset이 다시 나올 수 있다. 매번 **원래 입력**에서 선택 좌표를 교체하며, 이전 draw에서 변형된 입력을 누적해서 지우지 않는다. FE의 `features_in_step=1`은 모든 좌표를 한 번씩 독립적으로 교체한다. FE는 attribution을 먼저 정렬하지만, singleton에서는 paired observations의 순서만 바뀌므로 Pearson 값은 같다. [FC 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_correlation.py), [FE 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_estimate.py)

## 2. 실제로 어떻게 대조했나

`scripts/audit_quantus.py`는 다음을 실행한다.

1. 설치 버전과 8개 핵심 소스의 Git blob hash를 확인한다. 소스가 다르면 이 버전의 audit를 중단하고 실패를 기록한다.
2. `quantus.functions.perturb_func`의 실제 replacement 함수에 기록용 wrapper를 씌워 **Quantus가 직접 뽑은 subset**, 원래 입력, 교체된 입력을 보관한다. 계산은 원래 함수에 그대로 맡긴다.
3. `quantus.functions.similarity_func.correlation_pearson`에도 같은 방식으로 wrapper를 씌워 Quantus가 실제 전달한 `U`, `V`를 기록한다.
4. 그 **동일한 subset**을 프로젝트의 `perturbation_trace()`에 넣어 중간값과 `checked_pearson()` 점수를 비교한다.
5. NumPy adapter를 사용한 `evaluate_batch()`와 PyTorch scalar model을 사용한 전체 public `__call__()` 경로를 각각 점검한다. 후자는 preprocessing과 validation도 실행한다.

검증 fixture는 20개 feature, exact/A/B 세 설명, FC의 `k={1,5,10,19}`, `B=25`, 그리고 singleton FE이다. 정확한 이진 표현이 가능한 입력을 써서 PyTorch wrapper의 float32 입력 변환이 이 fixture의 값을 바꾸지 않도록 했다. 일반 실수 입력에서 모든 runtime이 비트 단위로 일치한다고 주장하지 않는다. Quantus의 RNG 함수, validation, 점수 코드 및 설치 파일은 수정하지 않았다. 기록 wrapper는 공식 함수에 그대로 위임하며, audit의 seed 설정은 종료 후 이전 NumPy RNG 상태로 복원된다.

### 관찰된 검증 결과

| 확인 항목 | 결과 |
|---|---|
| 실행 환경 | Python 3.12, Quantus 0.6.0, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu |
| source identity / utility / shape / parity 점검 | 34 / 34 통과; 실패 0, skip 0 |
| Quantus와 프로젝트의 `U`, `V` 최대 절대 차이 | 둘 다 `0.0` |
| score 최대 절대 차이 | `4.44e-16`; 허용 오차 `1e-12` 이내 |
| optional unittest | 5 / 5 통과 |
| 매 draw의 원래 입력 사용 | 확인; 누적 deletion 아님 |
| FE singleton과 coordinate Pearson | 지정 fixture에서 일치 |

저장된 실행 증거는 [quantus_audit.json](../results/verification/quantus_audit.json)에 있다. 이 표는 2026-11-04의 실제 실행을 요약한다. 새로 실행하면 기본적으로 `results/quantus_audit.json`을 만들어 보관된 증거를 덮어쓰지 않는다. package가 없을 때의 skip을 pass로 세지 않는다. PyTorch가 없으면 `evaluate_batch()` 대조만 가능하므로 보고서는 `partial`이며 public API는 미검증으로 남는다.

## 3. 실패·호환성 기록

### 재현한 upstream의 2-D FC 입력 오류

고정 버전의 `assert_value_smaller_than_input_size()`는 `(N,d)` 입력에 대해 feature 수를 확인한 뒤, 다시 `x.shape[2:]`의 곱과 비교한다. 이때 빈 shape의 곱은 `1`이므로 정상적인 `k=5,d=20`도 다음 오류를 낸다.

```text
'subset_size' must be smaller than input size. [5 >= 1.0]
```

전체 public FC 호출에서 이 실패를 재현했다. **라이브러리를 patch하지 않고 `(N,1,d)`로 channel 축을 추가**했으며, scalar model 안에서 다시 `(N,d)`로 펼쳤다. 동일한 `d`개 feature와 같은 scalar 함수가 유지되는 이 경로에서 FC와 FE가 통과했다. 이 결과는 upstream bug가 수정됐다는 뜻이 아니다. [validator 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/asserts.py), [PyTorch wrapper 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/model/pytorch_model.py)

### `"black"`은 숫자 0이 아니다

이 버전의 `"black"`은 각 입력의 최솟값이다. 실제 toy check에서 `[-2,3,4]`의 두 번째 좌표를 교체했을 때 숫자 `0.0`은 `[-2,0,4]`, `"black"`은 `[-2,-2,4]`를 만들었다. 따라서 EXP-01에는 literal zero를 전달한다. named preset을 숫자 zero로 가정하면 설명 reference와 evaluator baseline이 달라질 수 있다. [baseline 해석](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/utils.py), [replacement 구현](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/perturb_func.py)

### undefined와 문서 불일치

- Pearson에 상수 벡터를 넣으면 공식 similarity 함수는 warning과 `NaN`을 반환했다. 프로젝트도 이를 score 0으로 바꾸지 않고 invalid reason을 보존한다.
- FE parameter docstring은 Spearman이라고 쓰지만, 실행 코드는 Pearson을 기본값으로 설정한다. 이번 구현은 실행 코드 및 Atlas의 명시된 Pearson 계약을 따른다.
- 기존 pilot의 `n_scores=1`이나 누락된 `SensitivityN` 출력 원인은 이번 audit로 확정하지 않았다. 현재 defaults만으로 과거 결과를 역으로 진단할 수 없다.

## 4. utility 함수 이름을 해석할 때의 주의점

아래는 고정 소스와 설치된 함수 signature를 확인한 내용이다. EXP-01은 첫 번째 replacement와 Pearson만 사용하며, 다른 함수로 바꾸면 별도 metric variant 또는 다른 intervention이 된다.

| 함수 | 실제 역할과 적용 판단 |
|---|---|
| `batch_baseline_replacement_by_indices(arr, indices, perturb_baseline)` | `(N,F)` 입력의 선택 좌표를 baseline으로 교체한다. EXP-01의 내부 perturbation에 해당한다. |
| `baseline_replacement_by_indices(..., indexed_axes, ...)` | 비배치 입력에서 축을 추가로 지정한다. batch variant와 호출 형태가 다르다. |
| `batch_gaussian_noise`, `batch_uniform_noise` | 선택한 입력 좌표에 noise를 더한다. 고정 baseline deletion의 가정과 다르므로 EXP-01 결과에 섞지 않는다. |
| `no_perturbation` | 입력을 그대로 돌려준다. 모든 `V=0`이어서 이 FC 계약에서는 undefined control이 된다. |
| `noisy_linear_imputation` | 이웃 좌표와 공간 구조를 사용하는 구현이다. 순서만 있는 독립 tabular feature의 기본 교체법으로 채택하지 않았다. |
| `correlation_pearson`, `correlation_spearman`, `correlation_kendall_tau` | 각각 선형 association, 순위 association, 순서 일치 정도를 비교한다. 같은 값이나 의미를 보장하지 않는다. |
| `cosine` | 이름과 달리 `scipy.spatial.distance.cosine`을 반환한다. 동일 방향 벡터의 반환값은 `0`; 높은 점수가 더 좋은 similarity로 취급하면 안 된다. |
| `distance_euclidean`, `distance_manhattan` | 수치적 distance이다. 단위와 scale에 영향을 받으며 correlation의 대체 이름이 아니다. |
| `lipschitz_constant` | 분자의 변화량을 분모의 입력 변화량과 비교한다. 기본 norm도 서로 다르고 denominator에 `1e-10`을 더한다. FC의 두 벡터 comparator로 단순 교체할 함수가 아니다. |

[`perturb_func` 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/perturb_func.py), [`similarity_func` 소스](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/similarity_func.py)

## 5. 재실행

기본 환경에 optional dependencies를 추가한 다음 실행한다. 전체 public API 대조에는 PyTorch도 필요하며 GPU는 필요 없다. 아래 CPU 설치 명령은 Linux/Colab 환경 기준이다. 다른 운영체제는 PyTorch의 해당 플랫폼 설치 경로를 사용한다.

```bash
python -m pip install -r requirements-quantus.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python scripts/audit_quantus.py --output results/quantus_audit.json
python -m unittest discover -s tests -p test_quantus_parity.py -v
```

PyTorch 설치를 생략하면 utility와 직접 `evaluate_batch()` 대조만 실행되어 `partial`로 보고된다. audit는 확인용 fixture만 실행한다. 실제 EXP-01 실행·출력 해석·confirmation 순서는 [실험 설계 문서](EXP-01.md)에 따른다.
