# Kernel SKILL 文件索引

本目录存放用于 AI 辅助生成 EEGUnity kernel 的 SKILL prompt 模板。

使用方法：根据数据集的文件格式和标签存储方式选择对应 SKILL 文件，  
将其完整内容提供给 AI（Claude Code / Codex / 其他 LLM），  
同时提供数据集路径，AI 将自动探索数据集并生成 kernel。

---

## 选择 SKILL 文件

| 数据集特征 | 使用 SKILL 文件 |
|------------|-----------------|
| GDF / BDF / EDF，事件在 MNE Annotations 中 | [SKILL_gdf_annotation_remap.md](SKILL_gdf_annotation_remap.md) |
| `.mat` 文件（主文件或 sidecar），事件在 `.mat` 字段中 | [SKILL_mat_sidecar.md](SKILL_mat_sidecar.md) |
| EEGUnity 无法解析的私有格式 | [SKILL_custom_format.md](SKILL_custom_format.md) |
| BIDS 标准数据集（有 participants.tsv / events.tsv） | [SKILL_bids_standard.md](SKILL_bids_standard.md) |

## 决策树

```
数据集文件能被 MNE 直接读取？
├── 否 → SKILL_custom_format.md
└── 是 → 数据集遵循 BIDS 结构（有 participants.tsv）？
          ├── 是 → SKILL_bids_standard.md
          └── 否 → 事件标签在哪里？
                    ├── MNE Annotations（GDF/BDF/EDF header）→ SKILL_gdf_annotation_remap.md
                    └── 独立 .mat 文件字段 → SKILL_mat_sidecar.md
```

## 规范参考

所有 SKILL 文件的验证清单均对照 [KERNEL_RULEBOOK.md](../KERNEL_RULEBOOK.md)。  
生成的 kernel 应存放在 `app_kernel/kernels/` 目录中。
