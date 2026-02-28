# Release Benchmark Suite (v2)

Structured benchmark protocol for stability validation of `ml-leakage-guard`.

`ml-leakage-guard` 稳定性验证的结构化基准协议。

---

## English

### 1. Goal
- Validate **release stability**, not just single-run performance.
- Cover representative failure modes: leakage, overfit gaps, distribution shift, external transport, calibration/DCA, and fail-closed behavior.

### 2. Profiles (Registry-locked)
- `quick`
  - `authority_release_core` (WDBC + CKD stress route)
  - `adversarial_fail_closed`
- `release` (recommended)
  - `authority_release_core`
  - `authority_release_extended` (+ Diabetes130 large cohort, observational/non-blocking)
  - `adversarial_fail_closed`
- `extended`
  - `release` profile +
  - `authority_research_heart` (heart stress search; non-blocking by default unless `--heart-blocking`)
  
Release blocking suites are currently `authority_release_core` and `adversarial_fail_closed`.

Run command:

```bash
python3 scripts/mlgg.py benchmark-suite --profile release
```

Reproducibility hard gate (recommended):

```bash
python3 scripts/mlgg.py benchmark-suite \
  --profile release \
  --repeat 3 \
  --registry-file references/benchmark-registry.json \
  --emit-junit /tmp/mlgg_release_benchmark.junit.xml
```

### 3. Pass Contract
- Overall matrix passes only if **all blocking suites pass**.
- Registry must pass fingerprint lock checks (`benchmark_registry.v1`).
- Repeat consistency must pass (`repeat=3` by default).
- Non-blocking failures are still reported and must be reviewed before publication claims.
- Output report: `release_benchmark_matrix.v2`.
- Required fields:
  - `status_reason`
  - `failure_codes`
  - `repeat_count`
  - `repeat_consistent`
  - `dataset_registry_sha256`
  - `blocking_suite_ids`
  - `nonblocking_suite_ids`
- Non-blocking authority failures also emit:
  - `observational_diagnostics` (embedded in v2 report)
  - sidecar file `*.observational_diagnostics.json` (`release_benchmark_observational_diagnostics.v1`)

### 4. Why This Is Better Than "Many Random Datasets"
- Focuses on **risk coverage** instead of raw dataset count.
- Uses fixed routes and machine-readable summaries for reproducibility.
- Keeps strict fail-closed publication gates unchanged.

### 5. Core Dataset Coverage
- `uci-breast-cancer-wdbc`: small/clean baseline for full strict-chain sanity.
- `uci-chronic-kidney-disease`: missingness-heavy external stability route.
- `uci-diabetes-130-readmission`: larger heterogeneous cohort (runtime + transport stress, observational in release).
- `uci-heart-disease` (extended profile): high-pressure research stress route.

### 6. Standard Failure Codes
- `benchmark_registry_missing`
- `benchmark_registry_mismatch`
- `benchmark_repeat_inconsistent`
- `benchmark_blocking_suite_failed`

---

## 中文

### 1. 目标
- 验证的是**发布稳定性**，不是单次分数好看。
- 覆盖关键风险：泄漏、过拟合 gap、分布漂移、外部迁移、校准/DCA、fail-closed 行为。

### 2. 运行档位（registry 锁定）
- `quick`
  - `authority_release_core`（WDBC + CKD stress）
  - `adversarial_fail_closed`
- `release`（推荐）
  - `authority_release_core`
  - `authority_release_extended`（增加 Diabetes130 大样本，观测/非阻断）
  - `adversarial_fail_closed`
- `extended`
  - 在 `release` 基础上增加
  - `authority_research_heart`（heart 高压研究路线；默认非阻断，可用 `--heart-blocking` 设为阻断）

当前 release 阻断套件为 `authority_release_core` 与 `adversarial_fail_closed`。

执行命令：

```bash
python3 scripts/mlgg.py benchmark-suite --profile release
```

推荐可复现硬门：

```bash
python3 scripts/mlgg.py benchmark-suite \
  --profile release \
  --repeat 3 \
  --registry-file references/benchmark-registry.json \
  --emit-junit /tmp/mlgg_release_benchmark.junit.xml
```

### 3. 通过标准
- 仅当**所有阻断套件通过**时，矩阵整体 `pass`。
- 必须通过 registry 指纹锁定校验（`benchmark_registry.v1`）。
- 默认 `repeat=3`，重复结论不一致直接 fail。
- 非阻断失败不会直接拉闸，但会被写入报告并要求审查。
- 输出契约：`release_benchmark_matrix.v2`。
- 必填字段：
  - `status_reason`
  - `failure_codes`
  - `repeat_count`
  - `repeat_consistent`
  - `dataset_registry_sha256`
  - `blocking_suite_ids`
  - `nonblocking_suite_ids`
- 非阻断 authority 失败会额外输出：
  - `observational_diagnostics`（内嵌在 v2 报告）
  - sidecar 文件 `*.observational_diagnostics.json`（`release_benchmark_observational_diagnostics.v1`）

### 4. 为什么不是“数据集越多越好”
- 核心是**风险覆盖矩阵**，不是无序堆数据。
- 路线固定 + 机器可读报告，保证可复现可审计。
- 不放宽现有 publication-grade 严格门。

### 5. 数据集覆盖定位
- `uci-breast-cancer-wdbc`：小而稳的全链路基线。
- `uci-chronic-kidney-disease`：缺失值和外部稳健性压力。
- `uci-diabetes-130-readmission`：大样本异质性与迁移压力（release 中按观测项保留）。
- `uci-heart-disease`（extended）：高压研究路径验证。

### 6. 标准失败码
- `benchmark_registry_missing`
- `benchmark_registry_mismatch`
- `benchmark_repeat_inconsistent`
- `benchmark_blocking_suite_failed`
