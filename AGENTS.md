# AGENTS.md：给所有 AI 助手的项目说明

> Gemini、Codex、Cursor、Claude、Grok 等所有 AI 工具在本项目工作前，必须先读完本文件。

## 项目概述

设计 + AI 方向的论文选题与研究仓库。当前课题：**从图像生成到方案交付：OPC（一人公司）驱动的乡村农产品包装设计实践**，目标是一篇开放获取的 SCI/SSCI 期刊论文。

## 共享方法（开工前先读）

- 选题、文献树、判断创新点：
  https://github.com/Casey-sketch-cyber/findInnovation/blob/main/skills/research-innovation-points/SKILL.md
  模板：https://github.com/Casey-sketch-cyber/findInnovation/tree/main/templates
- 写论文故事：https://github.com/Casey-sketch-cyber/paperwrite
- 多 AI 协作流程：https://github.com/Casey-sketch-cyber/ai-workflow （`skills/agent-session/SKILL.md`）

## 目录结构

- `tasks/`：下达给各个 AI 的任务书，按编号排列。先看最新的、指派给你的那一份。
- `docs/`：文献树（`docs/literature-tree/`）、创新点评估（`INNOVATION.md`）、期刊调研（`JOURNALS.md`）、进度（`PROGRESS.md`）、决策（`DECISIONS.md`）。
- `literature/`：检索记录、文献清单。

## 研究约束

- 文献检索只用免费来源（作者没有 Scopus、Web of Science 权限）。
- 数据只用公开数据或自行网上采集的数据，不购买数据集。
- 不编造文献；引用必须打开原文或官方页面核对。

## 工作规则（所有 AI 必须遵守）

1. **开始前**：先读 `docs/PROGRESS.md`、`docs/DECISIONS.md` 和指派给你的任务书。
2. **分支与合并**：每个任务新建分支 `<工具名>/<任务简述>`（例如 `gemini/literature-tree-opc-packaging`），做完后推送并创建 PR；不能创建 PR 时只推送分支也可以。**不要自己合并到 `main`，也不用等用户**：Claude 审查后，通过就合并，没通过就在 PR 里留言并在 `docs/PROGRESS.md` 的待办里登记原因。开工前先看有没有自己之前被退回的 PR，先把它改完。不要强制推送，不要改写已推送的历史。
3. **范围**：只改与当前任务相关的文件。
4. **安全**：不要提交密钥、`.env`、大文件（论文 PDF 不要上传，只记链接）、AI 工具的缓存或对话记录。
5. **结束时**：更新 `docs/PROGRESS.md`（做了什么、没做完什么、遇到的坑）；重要取舍写进 `docs/DECISIONS.md`。
6. **沟通**：始终用中文。
