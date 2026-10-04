SourceBridge 研究代码与复现说明

代码按用途组织，脚本与模块使用正式语义名称，不包含编辑轮次或版本后缀。
论文中的方法名称、原始实验标识和随机种子命名空间保持一致；冻结 JSON、CSV、
NPZ 内记录的原始标识及哈希属于数据身份，不作为代码版本入口。

目录
  code/analysis       正文与附录误差指标的推导、充分统计量及完整结果。
  code/diagrams       四张概念图的可编辑矢量源、配色、图标与许可证。
  code/plots          八组实验图的绘图入口、冻结绘图数据及 PDF/SVG/PNG。
  code/experiments    完整实验结果表、冻结输入、科学实现、重放与计时入口。
  code/validation     代码完整性、导入与独立几何检查。

运行环境
  使用 Python 3.12 或更新版本。完整数据分析和科学重放依赖 NumPy、SciPy；
  表格指标及采购对照的独立重放使用标准库。绘图依赖 Matplotlib、Pillow、pypdf。
  依赖分别列于对应目录的 requirements.txt；本次整理未安装任何依赖。
  CVXPY/CLARABEL 仅供 group_methods.risk_weights 的可选风险权重例程使用，
  不影响本论文的主实验重放；该函数保留完整算法并在调用时导入可选依赖。
  使用 Times New Roman。Windows 自动检查标准字体目录；其他系统可设置
  SOURCEBRIDGE_FONT_DIR，指向包含 Times New Roman 字体文件的合法字体目录。

以下命令从本文件所在目录执行。各脚本以自身位置解析资源；也可以在其他
工作目录下通过脚本的绝对路径调用。不要移动代码内部的 data/source/fixtures。

主要复现入口
  python code/validation/verify_code.py
      核验发布代码和冻结输入哈希，解析所有 Python 源文件。

  python code/validation/check_imports.py
      导入 22 个科学 API 模块，不执行完整基准实验。

  python code/analysis/compute_error_metrics.py
      从充分统计量推导正文表格及补充误差指标。
  python code/analysis/compute_error_metrics.py --predictions-csv code/experiments/data/public_inference_cells.csv --output-dir code/experiments/generated/error_metrics
      同时独立核验全部 102,240 行原始预测。

  python code/experiments/analyze.py
      重新计算完整预测误差、1/3/7 日配对分块 bootstrap、20 种子采购区间，
      并检查 240 个预算决策及 480 个精度请求。输出到 experiments/generated。

  python code/experiments/replay.py
      实际执行 64 个冻结条件、15 个接口的 960 次预测及 64 个有理证书。
      包含两种参考配置、全部 16 个条件、首末观测目标。推断时不读取评价真值。

  python code/experiments/replay_procurement.py
      对全部 120 个候选池重算 JB-RobustD：7,680 个精确行列式、见证与驻点、
      1,932 个整数预算动作、240 个预算决策以及 480 个精度请求。

  python code/experiments/greedy_control.py
  python code/experiments/greedy_cleanup.py
      重算同目标贪心与可行删除控制。结果写到 experiments/generated。
  python code/experiments/audit_greedy.py
  python code/experiments/audit_cleanup.py
      使用独立标准库实现检查参考结果，保留全部有利与不利比较。

  python code/experiments/derive_plot_metrics.py
      从完整冻结记录重新汇总绘图数值与配对种子 bootstrap。
      默认生成 experiments/generated/plot_metrics.json，不覆盖冻结绘图输入。

  python code/experiments/run_timing.py verify
  python code/experiments/run_timing.py smoke
      分别验证 48 个已保存批次的 264 个查询结果，以及实际重算 n=8 目录与
      两种选择算法在预算 4/8 下的完整动作。
  python code/experiments/run_timing.py measure --output-dir PATH
      执行整个目录构建和查询批次计时；拒绝覆盖已有原始计时文件。
      时间依赖运行硬件，属于新测量，不应覆盖论文的冻结参考数据。

  python code/validation/check_geometry.py
      独立一维包络与有理三变量几何核对，包含方向支撑、见证、预算配对及投影。

图形复现
  python code/diagrams/regenerate.py
  python code/plots/plot_experiments.py
      分别生成概念图和实验图，并同步 manuscript_en/figures 与
      manuscript_zh/figures。绘图数据、原有区间、方法对应关系保持固定。
      catalog_reuse 是正文中的目录复用实验组。两种绘图入口均不生成实验预测。

科学实现与生成器
  experiments/source 保留原有的全部科学函数，而非仅提供绘图输出：
    reference_geometry / certified_geometry / packet_geometry / safe_projection
      事前宽度、精确证书、包络深度与安全投影。
    catalog_selection / additive_selection / robust_doptimal
      支撑目录、最小费用联合支撑与文献采购对照。
    public_inference / public_source_generation / acquisition_evaluation
      公开数据上的生成、购买、履约、推断与评价分离流程。
    weighted_inference / simulated_inference / mlni_calibration
      共同 WLS、机制诊断、参考先验 MLNI 与 proper 公共先验 MLNI。
    procurement / packet_attacks / attack_evaluation / selection_scaling
      候选池生成、固定攻击库、40 种子攻击实验及选择规模实验。
  本论文冻结记录中的 15 个接口均保留，包括 proper MLNI 与 SourceBridge
  重合的接口、参考先验变体、FETD-AK 及其他正文未重复绘出的对照。
  模块可执行动作由各文件的 --help 列出；完整生成会写 results/，建议在独立
  工作副本内执行。冻结协议已有完整 source_identity 核验，勿重新冻结替换。

数据范围与哈希
  experiments/data/public_inference_cells.csv 是完整 102,240 行预测表。
  experiments/data 还含完整 120 池输入/结果、64 子集宽度目录、所有费用请求、
  攻击条件结果与规模实验数据；experiments/fixtures 是声明的 64 条件回放子集。
  source/results/public_inference_extracted_data.npz 保存主实验的紧凑公开值；
  原始 55 MB MSRA Data-1.zip 不重复分发。重新执行原始归档提取时，须按冻结
  协议提供相同 SHA-256 的公开归档。588 MB 的全部求解器运行明细不重复打包；
  inference_timing_distribution.json 已含每个独立拟合的完整计时观察值。
  extract_inference_timings.py 可在提供原始明细时重新提取，并严格校验原哈希。

  预测CSV及科学实现原始字节保持不变。公开发布仅将5个JSON文件中的本机路径
  改为文件名或相对路径，全部数值保持不变，变更记录见PUBLICATION_PROVENANCE.json。
  代码名称、导入路径与说明规范化后，source_identity
  同时保留原记录哈希和当前发布代码哈希；verify_identity 同时检查二者，不将
  改名后的代码字节冒充原文件。resource_names 只解析冻结记录中的原资源名称。
  公开数据目标、攻击重复、参考配置之间的相关性，以及事后描述性汇总范围，
  均以论文和固定协议为准；重放通过不等同于新增独立实验或统计显著性。

第三方资源
  Lucide SVG 图标位于 code/diagrams/assets/lucide；原 LICENSE 和来源记录保留。
  文献方法的出处与场景适配见 source 中的正式方法说明及论文参考文献。
