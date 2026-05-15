# Skill 互相调用流程图

> 配套 `.github/skills/` 下四个 SKILL.md 的全局视图：
> [add-compat-api](../.github/skills/add-compat-api/SKILL.md) /
> [fix-compat-api](../.github/skills/fix-compat-api/SKILL.md) /
> [compatibility-testing](../.github/skills/compatibility-testing/SKILL.md) /
> [compat-doc-authoring](../.github/skills/compat-doc-authoring/SKILL.md)
>
> 配套源文件：[skill-flow.drawio](skill-flow.drawio)

## 总览（Mermaid）

```mermaid
flowchart TB
  %% 用户起点
  U1((用户输入\n新增 API 需求)):::entry
  U2((用户输入\nPR / Actions / review 链接)):::entry
  U3((用户输入\n算子名\n独立调用)):::entry
  U4((用户输入\n头文件名\n独立调用)):::entry

  %% add-compat-api
  subgraph add [add-compat-api - 驱动型上游]
    direction TB
    A1["Step 1\n确定本轮新增接口范围"]
    A2["Step 2\n参考 PyTorch + 新增 compat"]
    A2_4{{"Step 2 第 4 子项\n调用 compatibility-testing"}}:::invoke
    A3["Step 3\n编译 Paddle + ctest"]
    A4["Step 4\n安装新 wheel"]
    A5["Step 5\n复编 PCAT + result_cmp"]
    A6{"Step 6\n新增用例是否通过?"}:::decision
    A_DOC{{"文档归档\n调用 compat-doc-authoring"}}:::invoke
    A_DONE(["完成"]):::done

    A1 --> A2 --> A2_4 --> A3 --> A4 --> A5 --> A6
    A6 -- 否 --> A2
    A6 -- 是 --> A_DOC --> A_DONE
  end

  %% fix-compat-api
  subgraph fix [fix-compat-api - 驱动型上游]
    direction TB
    F0{"链接分流\nPR / Actions / review?"}:::decision
    F0_PR["分支 A\nPR 链接"]
    F0_AC["分支 B\nActions 链接"]
    F0_RV["分支 C\nreview/comment 链接"]
    F1["Step 1\n解析需求并定义目标"]
    F_EXIT{{"横向退出\n根因为接口完全缺失"}}:::invoke
    F2["Step 2\n对照 PyTorch 实现"]
    F3["Step 3\n在 Paddle 侧最小修复"]
    F3_3{{"Step 3 第 3 子项\n调用 compatibility-testing"}}:::invoke
    F4["Step 4\n编译 + ctest"]
    F5["Step 5\nwheel + result_cmp"]
    F6{"Step 6\n问题是否消失?"}:::decision
    F_DOC{{"文档归档\n调用 compat-doc-authoring"}}:::invoke
    F_DONE(["完成"]):::done

    F0 --> F0_PR
    F0 --> F0_AC
    F0 --> F0_RV
    F0_PR --> F1
    F0_AC --> F1
    F0_RV --> F1
    F1 --> F_EXIT
    F1 --> F2 --> F3 --> F3_3 --> F4 --> F5 --> F6
    F6 -- 否 --> F2
    F6 -- 是 --> F_DOC --> F_DONE
  end

  %% compatibility-testing
  subgraph test [compatibility-testing - 规范型下游]
    direction TB
    T_IN[\输入：算子名 / 覆盖目标\n输出路径 / PCAT_ROOT/]:::callin
    T1["读约定\nShape 四档 + Dtype 四基础类型"]
    T2["按 OpName 生成测试骨架\nwrite_op_result_to_file"]
    T3["覆盖\nShape/Dtype/值域/API/异常"]
    T_CHK{"新算子 checklist\n强制项全过?"}:::decision
    T_OUT[\输出：测试文件骨架\n+ checklist 自检结论/]:::callout

    T_IN --> T1 --> T2 --> T3 --> T_CHK
    T_CHK -- 否 --> T3
    T_CHK -- 是 --> T_OUT
  end

  %% compat-doc-authoring
  subgraph docsub [compat-doc-authoring - 规范型下游]
    direction TB
    D_IN[\输入：调用模式 / 目标文档\n上游模板名 / 已填段落/]:::callin
    D_MODE{"调用模式?"}:::decision
    D_STD["独立调用\n按标准模板新建"]
    D_APPEND["append-to-existing\n原样追加 + 改表"]
    D1["Step 1 读代码列 API 清单"]
    D2["Step 2 分组 + 标 ✅🔧❌"]
    D3["Step 3 写入文档"]
    D4["Step 4 兼容性统计回填"]
    D5{"Step 5 校验\n9 项 checklist 全过?"}:::decision
    D_OUT[\输出：归档结论\n✅n / 🔧n / ❌n/]:::callout

    D_IN --> D_MODE
    D_MODE -- 独立 --> D_STD --> D1 --> D2 --> D3 --> D4 --> D5
    D_MODE -- append --> D_APPEND --> D4
    D5 -- 否 --> D3
    D5 -- 是 --> D_OUT
  end

  %% 入边
  U1 ==> A1
  U2 ==> F0
  U3 ==> T_IN
  U4 ==> D_IN

  %% 跨 skill 实线调用
  A2_4 ==>|"PCAT_ROOT 算子名 覆盖目标"| T_IN
  F3_3 ==>|"PCAT_ROOT 算子名 修复点"| T_IN
  A_DOC ==>|"append-to-existing 已填段落"| D_IN
  F_DOC ==>|"append-to-existing 已填段落"| D_IN

  %% 跨 skill 回流
  T_OUT -.->|"骨架就绪"| A3
  T_OUT -.->|"增量就绪"| F4
  D_OUT -.->|"归档校验通过"| A_DONE
  D_OUT -.->|"归档校验通过"| F_DONE

  %% 横向说明性引用
  F_EXIT -. "说明性引用\n请退出 fix 改用 add" .-> A1

  %% 样式
  classDef entry fill:#ffe9b3,stroke:#cc8800,stroke-width:2px
  classDef driver fill:#e3f2fd,stroke:#1565c0
  classDef spec fill:#f3e5f5,stroke:#6a1b9a
  classDef decision fill:#fff3cd,stroke:#856404
  classDef invoke fill:#d4edda,stroke:#155724
  classDef callin fill:#cfe2ff,stroke:#084298
  classDef callout fill:#cfe2ff,stroke:#084298
  classDef done fill:#d1e7dd,stroke:#0f5132,stroke-width:2px

  %% 边颜色（与 draw.io 保持一致）
  linkStyle 46 stroke:#1565c0,stroke-width:3px
  linkStyle 47 stroke:#2e7d32,stroke-width:3px
  linkStyle 48 stroke:#6a1b9a,stroke-width:3px
  linkStyle 49 stroke:#e65100,stroke-width:3px
  linkStyle 54 stroke:#842029,stroke-width:2px
  linkStyle 50 stroke:#1565c0,stroke-width:2px
  linkStyle 51 stroke:#2e7d32,stroke-width:2px
  linkStyle 52 stroke:#6a1b9a,stroke-width:2px
  linkStyle 53 stroke:#e65100,stroke-width:2px
```

## 图例

| 元素 | 含义 |
|------|------|
| 圆角矩形 `((U1))` | 用户起点（每个 skill 都可作为起点） |
| 方框 `[Step N]` | skill 内部 Step |
| 菱形 `{判定}` | 条件分支或循环判定 |
| 六边形 `{{跨 skill 调用点}}` | 触发跨 skill 调用的子步骤 |
| 平行四边形 `[/IO/]` | 下游 skill 的输入/输出契约 |
| 椭圆 `([完成])` | skill 终态 |
| **粗实线** `==>` | 自动跳转（Claude 在该节点会发起 Skill 工具调用） |
| 细虚线 `-.->` | 回流路径（下游产出后续接到调用方下一步） |
| 标注 `说明性引用` 的虚线 | **人工切换**（不自动触发，避免双驱动循环递归） |

## 起点与回流速查

- **U1（新增 API 需求）** → add-compat-api: A1 → A2 → A2_4 ⟹ T_IN ⟶ T_OUT ⟼ A3 → A4 → A5 → A6 ⟶ A_DOC ⟹ D_IN ⟶ D_OUT ⟼ A_DONE
- **U2（PR / Actions / review 链接）** → fix-compat-api: F0 → F1 → F2 → F3 → F3_3 ⟹ T_IN ⟶ T_OUT ⟼ F4 → F5 → F6 ⟶ F_DOC ⟹ D_IN ⟶ D_OUT ⟼ F_DONE
- **U2 触发横向退出**（接口完全缺失）→ fix-compat-api: F0 → F1 → F_EXIT ⤳ A1（**人工切换**）
- **U3（独立写测试）** → compatibility-testing: T_IN → T1 → T2 → T3 → T_CHK → T_OUT（无跨 skill 调用）
- **U4（独立写文档）** → compat-doc-authoring: D_IN → D_MODE(独立) → D_STD → D1 → ... → D5 → D_OUT（无跨 skill 调用）

## 防循环约束（图未画出但规则强制）

- `compatibility-testing` 与 `compat-doc-authoring` 节点**只接收上游传入、产出后回流，绝不主动指向 add/fix 内部节点**
- `F_EXIT → A1` 是图上唯一的横向跨 skill 边，且标注为"说明性引用"，靠人工/Claude 显式切换，不发起 `Skill(skill=..., ...)` 工具调用
- 下游 skill 的 SKILL.md 在"上游调用上下文"段已显式声明 "**不反向调用任何上游 skill，也不触发其他 skill**"，从规则层面与图拓扑层面双重切断循环路径

## draw.io 版本

源文件：[skill-flow.drawio](skill-flow.drawio)

打开方式：

- VSCode：安装 [Draw.io Integration](https://marketplace.visualstudio.com/items?itemName=hediet.vscode-drawio) 插件后直接双击 `.drawio`
- 在线编辑：上传至 [https://app.diagrams.net](https://app.diagrams.net)
- 导出：建议导出为 `skill-flow.svg`（`File → Export As → SVG`），便于 README / Wiki 嵌入

## 维护约束

1. 修改 4 个 SKILL.md 的 Step 划分或跨 skill 调用边时，**必须同步**更新 `skill-flow.md` 与 `skill-flow.drawio`
2. 不在 SKILL.md 内嵌任何 Mermaid 代码块——所有流程图维护点收敛在本文件
3. 新增 skill 时，先在本流程图中加节点和边、再去新建 `.github/skills/<new-skill>/SKILL.md`，避免拓扑漂移
4. 横向跨 skill 边只允许"说明性引用"（虚线 + `请退出 ... 改用 ...` 文字），禁止自动 `Skill(skill=..., ...)` 调用，防止双驱动循环递归
