# ML-Leakage-Guard 安全加固 Skill 归纳总结

> 加固日期: 2026-03-09  
> 加固范围: 全项目安全防御体系  
> 涉及文件: 15+ 文件修改/新增  
> 测试覆盖: 152 个安全专项测试

---

## 一、加固全景图

| 层级 | 威胁 | 防御措施 | 实现文件 | 测试文件 |
|------|------|----------|----------|----------|
| L0 反序列化 | Pickle RCE（任意代码执行） | `RestrictedUnpickler` 白名单沙盒 | `_security.py` | `test_security_deep.py` |
| L1 模型签名 | 模型工件篡改 | HMAC-SHA256 签名 + `.pkl.sig` 侧车文件 | `_security.py` | `test_security.py` |
| L2 证据完整性 | 证据文件被修改 | SHA256 清单 `.manifest.json` | `_security.py` | `test_security.py` |
| L3 静态加密 | 敏感数据泄露（磁盘级） | AES-256-GCM + PBKDF2 降级 | `_security.py` | `test_security_deep.py` |
| L4 审计日志 | 日志篡改/删除 | HMAC-SHA256 链式日志 `.gate_audit.jsonl` | `_gate_utils.py` | `test_audit_logging.py` |
| L5 路径防护 | 路径穿越注入 | null byte 拒绝 + 禁止系统路径 + 沙盒 | `_gate_utils.py` | `test_audit_logging.py` |
| L6 输入验证 | CLI 参数注入 | 长度限制 4096 + null byte 拒绝 | `_gate_framework.py` | `test_security_deep.py` |
| L7 资源耗尽 | DoS 内存/磁盘 | JSON 100MB / CSV 2GB 文件大小限制 | `_gate_utils.py`, `split_data.py`, `train_select_evaluate.py` | — |
| L8 Web 安全 | XSS/Clickjacking/DDoS | CSP + X-Frame + nosniff + Rate Limit 30/min | `mlgg_web.py` | — |
| L9 上传防护 | 恶意文件上传 | 文件名正则清洗 + 扩展名白名单(.csv) | `mlgg_web.py` | — |
| L10 文件清理 | 数据恢复攻击 | `secure_delete` 零填充后删除 | `_security.py` | `test_security_deep.py` |
| L11 供应链 | 依赖篡改 | `verify_critical_imports` 运行时检查 | `_security.py` | `test_security.py` |
| L12 敏感扫描 | 凭据泄露到证据 | 复合关键词模式匹配（避免误报） | `security_audit_gate.py` | `test_security_audit_gate.py` |
| L13 权限控制 | 未授权操作 | RBAC 4 角色：admin/operator/auditor/viewer | `_security.py` | `test_rbac_receipts.py` |
| L14 不可否认 | 执行结果伪造 | HMAC 签名执行回执 `.execution_receipt.json` | `_security.py` | `test_rbac_receipts.py` |
| L15 Gate 审计 | 安全检查遗漏 | 第 29 道 gate `security_audit_gate` | `security_audit_gate.py` | `test_security_audit_gate.py` |
| L16 管线自动加密 | 证据完成后未加密 | `--encrypt` flag 自动 AES 加密所有 JSON | `run_dag_pipeline.py` | — |
| L17 管线执行回执 | 执行结果伪造/否认 | `--sign-receipt` flag 自动生成 HMAC 签名回执 | `run_dag_pipeline.py` | `test_rbac_receipts.py` |
| L18 管线安全清理 | 临时文件残留 | `--secure-cleanup` flag 零填充删除 *.tmp/*.log | `run_dag_pipeline.py` | — |
| L19 管线 RBAC 入口 | 未授权执行管线 | `--require-role` flag 检查用户权限 | `run_dag_pipeline.py` | `test_rbac_receipts.py` |
| L20 CSRF 防护 | 跨站请求伪造 | 每次渲染表单生成 token，POST 验证后销毁 | `mlgg_web.py` | — |

---

## 二、遇到的错误与修复记录

### 错误 1: `print_gate_summary` 调用签名不匹配

- **症状**: `security_audit_gate.py` 测试时 `TypeError: print_gate_summary() got unexpected keyword argument`
- **根因**: `_gate_framework.py` 中 `print_gate_summary` 函数签名是位置参数，不是关键字参数
- **修复**: 改用位置参数调用 `print_gate_summary(GATE_NAME, status, failures, warnings, args.strict, get_gate_elapsed())`
- **教训**: **调用框架函数前务必先读取函数签名**，不要凭记忆假设参数形式

### 错误 2: Gate 计数测试断言失败 (28 → 29)

- **症状**: `test_expected_gate_count` 断言 `len(GATE_REGISTRY) == 28` 失败
- **根因**: 新增 `security_audit_gate` 后注册表有 29 个 gate，但测试硬编码了 28
- **修复**: 更新断言为 `== 29`
- **教训**: **新增 gate 后必须全局搜索所有硬编码的 gate 计数**，包括测试和文档

### 错误 3: `self_critique_gate` 依赖检查失败

- **症状**: `test_self_critique_depends_on_all_gates` 断言 `security_audit_gate` 应在 `self_critique_gate.depends_on` 中
- **根因**: `self_critique_gate` 通过 `frozenset(name for name in GATE_REGISTRY)` 注册依赖，但 `security_audit_gate` 是在它**之后**注册的，不在其依赖集中。同时 `security_audit_gate` 反向依赖 `self_critique_gate`
- **修复**: 修改测试排除"依赖 self_critique 的 gate"（即后置 gate），不要求它们被 self_critique 依赖
- **教训**: **DAG 中添加新的尾节点时，必须理解依赖方向**。注册顺序决定了 `frozenset(name for name in GATE_REGISTRY)` 的内容

### 错误 4: `test_self_critique_is_last` 失败

- **症状**: 拓扑排序最后一个 gate 不再是 `self_critique_gate` 而是 `security_audit_gate`
- **根因**: `security_audit_gate` 依赖 `self_critique_gate`，所以在拓扑排序中排最后
- **修复**: 测试改为断言 `order[-1] == "security_audit_gate"`
- **教训**: **新增管线尾部 gate 必须更新拓扑排序相关的所有测试断言**

### 错误 5: `test_get_dependents_leaf_gate` 失败

- **症状**: `get_dependents("self_critique_gate")` 返回非空集合
- **根因**: `security_audit_gate` 依赖 `self_critique_gate`，所以 self_critique 不再是叶子节点
- **修复**: 改为检查真正的叶子节点 `get_dependents("security_audit_gate")`
- **教训**: **叶子节点测试必须跟随 DAG 尾部变化更新**

### 错误 6: Gate 状态使用非标准值 `"warn"`

- **症状**: 未触发测试失败，但 `publication_gate` 聚合时可能误判
- **根因**: `security_audit_gate.py` 中 `status = "fail" if failures else ("warn" if warnings else "pass")`，`"warn"` 不是标准状态
- **修复**: 改为二元状态 `status = "fail" if failures else "pass"`
- **教训**: **Gate 框架只接受 `pass`/`fail` 两种状态**，warnings 通过 report 中的 `warnings` 字段传递，不影响 status

### 错误 7: 敏感数据 Pattern 误报

- **症状**: `"token"` 匹配了 `"tokenizer"`（合法 ML 术语），`"secret"` 匹配了 `"secret_garden"` 等
- **根因**: 使用了过于宽泛的单词 `"secret"` 和 `"token"` 作为匹配模式
- **修复**: 改用复合模式 `"secret_key"`, `"auth_token"`, `"bearer_token"`, `"api_secret"` 等
- **教训**: **敏感数据扫描 pattern 必须足够具体**，避免在 ML 语境中产生误报。同时需要在 `_security.py` 和 `security_audit_gate.py` 两处保持一致

### 错误 8: 手工构造 Pickle Payload 格式错误

- **症状**: `test_blocks_os_system` 等测试报 `pickle data was truncated`
- **根因**: 手工编写的二进制 pickle payload 字节序和长度不正确
- **修复**: 改用 pickle 模块的常量（`pickle.GLOBAL`, `pickle.REDUCE`, `pickle.STOP` 等）程序化构造
- **教训**: **不要手写二进制 pickle payload**，使用 pickle 模块提供的 opcode 常量组合构造。格式为 `PROTO + GLOBAL(module\nfunc\n) + arg + TUPLE1 + REDUCE + STOP`

### 错误 9: SecureModelLoader 直接用 joblib.load 绕过沙盒

- **症状**: 虽然实现了 `RestrictedUnpickler`，但 `SecureModelLoader.load()` 仍然调用 `joblib.load()` 而非受限加载器
- **根因**: `RestrictedUnpickler` 和 `SecureModelLoader` 是不同阶段开发的，未衔接
- **修复**: 改为先尝试 `safe_pickle_load(fh)`，仅在非 SecurityError 异常时（如 joblib 压缩格式）降级到 `joblib.load()`
- **教训**: **实现安全组件后必须检查所有调用点是否已集成**。安全组件如果不在实际执行路径上就等于没有

### 错误 10: `_finalize` 安全后处理需要 best-effort 模式

- **症状**: 如果加密或签名环节失败，不应阻止管线正常返回结果
- **根因**: 安全后处理是可选增强功能，不是管线的核心判定逻辑
- **修复**: 用 `try/except` 包裹加密和签名逻辑，失败时仅打印 `[WARN]`，不改变 exit code
- **教训**: **安全增强功能应该 best-effort**，不能因为增强功能失败而破坏核心功能

---

## 三、关键设计决策与理由

### 3.1 RestrictedUnpickler 白名单 vs 黑名单

- **决策**: 采用白名单（只允许 sklearn/numpy/scipy/pandas/joblib 模块）
- **理由**: 黑名单无法穷举所有危险模块（如 `nt.system`, `posix.system`, 自定义恶意模块）。白名单虽然需要维护，但安全性更高
- **注意**: 当升级 sklearn 版本引入新子模块时，可能需要扩展 `_ALLOWED_PICKLE_MODULES`

### 3.2 AES-256-GCM + PBKDF2 降级

- **决策**: 优先使用 `cryptography` 库的 AESGCM，不可用时降级为 PBKDF2+XOR+HMAC
- **理由**: `cryptography` 不是必须依赖（仅 `[web]` extras 安装），降级模式仍提供机密性和完整性
- **注意**: 降级模式的 XOR 流加密 **不等同于** AES 安全性，生产环境应安装 `cryptography`

### 3.3 审计日志 HMAC 链

- **决策**: 每条日志用前一条的 `chain_hash` 作为 HMAC 密钥，形成单向链
- **理由**: 任何中间条目被修改或删除，后续所有条目的哈希都会失效，立即检出篡改
- **注意**: 第一条记录使用 `"0" * 64` 作为初始哈希，这是公开值不影响安全性

### 3.4 Gate 状态只用 pass/fail

- **决策**: 去掉 `"warn"` 状态，只使用二元 `"pass"` / `"fail"`
- **理由**: `publication_gate` 和 `self_critique_gate` 聚合所有 gate 结果时，只检查 `"pass"` 和 `"fail"`。非标准状态会导致聚合逻辑混乱
- **注意**: Warnings 通过 report JSON 中的 `"warnings"` 数组传递，不影响 gate 的 pass/fail 判定

### 3.5 RBAC 默认角色为 viewer

- **决策**: 未注册用户默认分配 `viewer` 角色（只读）
- **理由**: 最小权限原则。未知用户不应有写入或执行权限
- **注意**: 首次部署时管理员需要手动 `assign_role(username, Role.ADMIN)`

### 3.6 管线安全后处理用 best-effort 模式

- **决策**: `--encrypt` 和 `--sign-receipt` 在 `_finalize` 中用 `try/except` 包裹
- **理由**: 安全增强是可选层，不应阻塞管线核心判定（pass/fail）和报告输出
- **注意**: 如果加密失败，证据仍以明文存在，应在 CI/CD 中检查 `.enc` 文件是否生成

### 3.7 SecureModelLoader 使用「先沙盒后降级」策略

- **决策**: 先尝试 `RestrictedUnpickler`，仅在格式不兼容（如 joblib 压缩）时降级 `joblib.load`
- **理由**: joblib 使用 zlib/lz4 压缩后的文件不是标准 pickle 格式，`RestrictedUnpickler` 无法直接解析
- **注意**: 降级路径仍然存在理论上的 RCE 风险，但受 HMAC 签名验证保护（未签名/签名不匹配的模型直接拒绝加载）

---

## 四、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `scripts/security_audit_gate.py` | 新增 | 第 29 道 gate：安全审计 |
| `scripts/_security.py` | 修改 | RestrictedUnpickler, AES 加密, RBAC, 执行回执, 安全删除, 新 CLI 命令 |
| `scripts/_gate_utils.py` | 修改 | JSON 大小限制, 审计日志, resolve_path 加固 |
| `scripts/_gate_framework.py` | 修改 | CLI 参数长度验证 |
| `scripts/_gate_registry.py` | 修改 | 注册 security_audit_gate |
| `scripts/run_dag_pipeline.py` | 修改 | 审计日志集成, security_audit_gate 命令构建, --encrypt/--sign-receipt/--secure-cleanup/--require-role 安全后处理 |
| `scripts/run_strict_pipeline.py` | 修改 | --encrypt/--sign-receipt/--secure-cleanup 安全后处理对齐 |
| `scripts/mlgg_web.py` | 修改 | CSP headers, rate limiting, 文件名清洗, 路径防护 |
| `scripts/split_data.py` | 修改 | CSV 大小限制 |
| `scripts/train_select_evaluate.py` | 修改 | CSV 大小限制, categorical_analysis 嵌入 |
| `tests/test_security_audit_gate.py` | 新增 | 13 个 gate 测试 |
| `tests/test_audit_logging.py` | 新增 | 14 个审计日志测试 + 5 个路径防护测试 |
| `tests/test_security_deep.py` | 新增 | 24 个深层安全测试 |
| `tests/test_rbac_receipts.py` | 新增 | 18 个 RBAC + 执行回执测试 |
| `tests/test_gate_registry.py` | 修改 | 更新断言适配 29 gate |
| `README.md` | 修改 | 安全章节扩展 |
| `SKILL.md` | 修改 | Gate 计数、序列、输出合约更新 |
| `.gitignore` | 修改 | 新增安全工件排除项 |

---

## 五、CLI 命令速查

```bash
# 模型签名/验证
python3 scripts/_security.py sign models/model.pkl
python3 scripts/_security.py verify models/model.pkl

# 证据清单
python3 scripts/_security.py manifest evidence/

# 安全审计
python3 scripts/_security.py audit evidence/

# 依赖完整性
python3 scripts/_security.py check-deps

# 证据加密/解密
python3 scripts/_security.py encrypt evidence/
python3 scripts/_security.py decrypt evidence/report.json.enc

# 安全删除
python3 scripts/_security.py secure-delete evidence/ --pattern "*.tmp"

# 审计日志链验证
python3 scripts/_security.py verify-audit evidence/

# 管线安全 flags（可组合使用）
python3 scripts/run_dag_pipeline.py --request request.json --strict \
  --encrypt --sign-receipt --secure-cleanup --require-role operator
```

---

## 六、常见陷阱速查（从错误记录提炼）

| 陷阱 | 触发场景 | 规避方法 |
|------|----------|----------|
| 硬编码计数 | 新增/删除 gate 后忘记更新 | `grep -rn "28\|29" tests/ README.md SKILL.md` |
| DAG 尾节点变更 | 新增尾部 gate 后叶子节点/拓扑排序测试失败 | 检查所有 `order[-1]`、`get_dependents` 测试 |
| 非标准 gate 状态 | 使用 `"warn"` 等非二元状态 | 只用 `"pass"`/`"fail"`，warnings 放 report 数组 |
| 宽泛敏感 pattern | `"token"` 匹配 `"tokenizer"` | 用复合词 `"auth_token"` 代替单词 `"token"` |
| 安全组件未集成 | 实现了沙盒但实际调用点仍用旧方法 | 实现后立即 grep 所有调用点确认替换 |
| 手写二进制协议 | pickle/protobuf payload 字节错误 | 用库提供的常量/builder 构造 |
| 增强功能阻塞核心 | 加密失败导致管线报错 | 增强层用 `try/except` + `[WARN]` best-effort |
| 多处定义同一 pattern | `_security.py` 和 `security_audit_gate.py` 敏感词不一致 | 抽取为共享常量或单一来源（已修复→`SENSITIVE_DATA_PATTERNS`） |

---

## 七、Commit 记录

| SHA | 说明 |
|-----|------|
| `51b8510` | security_audit_gate + JSON/CSV 大小限制 + gate 状态修复 + 文档更新 |
| `83fd2c3` | 审计日志 + Web 加固 + 路径防护 |
| `c10b01e` | 敏感 pattern 对齐 + .gitignore |
| `591a816` | RestrictedUnpickler + AES 加密 + 安全删除 + Rate Limiting |
| `b59918c` | CLI 新命令 + README 安全文档扩展 |
| `051e545` | RBAC + 签名执行回执 + RestrictedUnpickler 集成到 SecureModelLoader |
| `3d68469` | 安全加固 Skill 归纳总结文档 |
| `a999c05` | --encrypt/--sign-receipt 管线集成 + Skill 文档更新 |
| `f247b45` | --secure-cleanup/--require-role 管线集成 + Skill L18-L19 |
| `ee85885` | README English security section (6.1) |
| `430d3bd` | Skill doc commit log 最终化 |
| `dfb9e67` | run_strict_pipeline 安全 flags + DRY 敏感 pattern + Skill 陷阱表 |
| `(pending)` | CSRF token 防护 + L20 + Skill 最终更新 |

---

## 七、后续可选加固方向

1. **mTLS 通信**: 如果 mlgg_web.py 需要对外暴露，添加双向 TLS 认证
2. **HSM 密钥管理**: 将 HMAC/AES 密钥存入硬件安全模块（如 AWS KMS/Azure Key Vault）
3. **WAF 集成**: Web 应用防火墙规则（SQL 注入、XSS payload 检测）
4. **SIEM 集成**: 审计日志推送到 Splunk/ELK 等安全信息和事件管理平台
5. **SBOM 生成**: 软件物料清单（Software Bill of Materials），用于供应链透明度
6. **容器沙盒**: 在 gVisor/Kata 容器中运行 gate，隔离文件系统
