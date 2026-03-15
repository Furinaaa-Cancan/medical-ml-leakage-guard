# Papers — MLGG Framework Validation Library

本目录用于收录医疗 ML 预测/分类文章，通过 MLGG 框架对其进行量化审查，验证框架的检测能力和覆盖范围。

---

## 目录结构

```
papers/
├── README.md                          # 本文件
├── .gitignore                         # 忽略 PDF 文件（可选上传）
│
├── templates/
│   └── paper_metadata_template.json   # 元数据填写模板
│
├── manifests/                         # batch_journal_review.py 的输入清单
│   ├── batch_manifest_all.json        # 全量审查
│   ├── batch_manifest_nature_medicine.json
│   ├── batch_manifest_lancet_dh.json
│   └── batch_manifest_jama_bmj.json
│
├── <journal>/                         # 第一维：期刊等级
│   └── <disease>/                     # 第二维：疾病领域
│       └── <first_author_year_keyword>/   # 每篇文章一个文件夹
│           ├── paper.pdf              # PDF 原文（.gitignore 忽略）
│           ├── metadata.json          # 结构化元数据（必填）
│           └── audit_output/          # MLGG 审查输出（自动生成）
│
└── audit_results/                     # batch 审查汇总结果
```

### 期刊分类（第一层）

| 目录名 | 对应期刊 | MLGG target_journal 值 |
|--------|---------|----------------------|
| `nature_medicine/` | Nature Medicine | `nature_medicine` |
| `lancet_digital_health/` | Lancet Digital Health | `lancet_digital_health` |
| `jama/` | JAMA、JAMA Internal Medicine 等 | `jama` |
| `bmj/` | BMJ、BMJ Open 等 | `bmj` |
| `npj_digital_medicine/` | npj Digital Medicine | `npj_digital_medicine` |
| `specialist_journals/` | 其余专科期刊 | — |

### 疾病分类（第二层）

| 目录名 | 包含范围 |
|--------|---------|
| `cardiovascular/` | 房颤、心衰、心梗、卒中风险等 |
| `oncology/` | 癌症诊断、复发预测、治疗响应等 |
| `diabetes/` | T2DM 诊断、并发症风险、血糖控制等 |
| `kidney_disease/` | AKI、CKD 进展、透析预测等 |
| `sepsis_icu/` | 脓毒症、ICU 死亡率、器官衰竭等 |
| `neurology/` | 痴呆、癫痫、神经退行等 |
| `respiratory/` | COPD、哮喘、肺炎、COVID-19 等 |
| `infectious_disease/` | 感染性疾病（非呼吸道） |
| `other/` | 其他不属于以上类别的 |

---

## 添加新文章

### 步骤 1：创建文章目录

命名规则：`<first_author_lastname>_<year>_<2-3_keyword>`，全小写，用下划线。

```bash
mkdir -p papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction
```

### 步骤 2：复制并填写元数据模板

```bash
cp papers/templates/paper_metadata_template.json \
   papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction/metadata.json
# 然后编辑 metadata.json，填写文章信息
```

**必填字段**：
- `bibliographic.title`, `authors`, `journal`, `year`, `doi`
- `study_design.outcome`, `prediction_type`
- `dataset.source_type`, `n_patients_total`, `n_events_positive`
- `model.model_type`
- `performance_metrics.test_auroc`
- `leakage_risk_assessment.*`（逐项根据 Methods 节填写）
- `reviewer_notes.added_by`, `added_date`

### 步骤 3：放入 PDF（可选）

```bash
cp /path/to/paper.pdf papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction/paper.pdf
```

PDF 已在 `.gitignore` 中忽略（版权保护）。如需共享，请上传至私有存储后在 `metadata.json` 的 `bibliographic.url` 填写链接。

### 步骤 4：更新 manifest

在对应的 `manifests/batch_manifest_*.json` 的 `projects` 数组中追加：

```json
{
  "id": "smith_2023_af_ehr_prediction",
  "path": "papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction",
  "label": "Smith et al. 2023 — AF prediction from EHR (Nature Medicine)",
  "notes": "External validation on 2 cohorts. TRIPOD+AI claimed."
}
```

---

## 运行审查

### 审查单篇文章

```bash
python3 scripts/audit_external_project.py \
  --project-dir papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction \
  --target-journal nature_medicine \
  --output papers/nature_medicine/cardiovascular/smith_2023_af_ehr_prediction/audit_output/audit_report.json
```

### 批量审查某期刊所有文章

```bash
python3 scripts/batch_journal_review.py \
  --manifest papers/manifests/batch_manifest_nature_medicine.json \
  --target-journal nature_medicine \
  --output papers/audit_results/batch_nature_medicine.json \
  --format markdown \
  --summary-csv papers/audit_results/batch_nature_medicine_summary.csv \
  --workers 4
```

### 全量批量审查

```bash
python3 scripts/batch_journal_review.py \
  --manifest papers/manifests/batch_manifest_all.json \
  --output papers/audit_results/batch_all.json \
  --format markdown \
  --summary-csv papers/audit_results/batch_summary.csv
```

---

## 审查结果解读

| MLGG 总分 | 等级 | 含义 |
|----------|------|------|
| 90–100 | Publication-grade | 方法论无重大缺陷 |
| 75–89 | Solid but gaps remain | 存在可修复缺口 |
| 60–74 | Major issues | 重大方法论问题 |
| <60 | Not publishable | 不应作为方法论参考 |

---

## 当前收录统计

> 更新脚本：`python3 scripts/audit_external_project.py --count-papers papers/`（待实现）

| 期刊 | 心血管 | 肿瘤 | 糖尿病 | 肾病 | 脓毒症/ICU | 神经 | 呼吸 | 感染 | 其他 | 合计 |
|------|--------|------|--------|------|-----------|------|------|------|------|------|
| Nature Medicine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Lancet Digital Health | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| JAMA | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BMJ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| npj Digital Medicine | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 专科期刊 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **合计** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
