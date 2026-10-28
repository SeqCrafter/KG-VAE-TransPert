# 本地源码复用记录

原仓库保持不变。复制的许可文件位于 `kgvae/vendor/MAP_LICENSE` 和 `kgvae/vendor/WAVE_LICENSE`。

| 本项目文件 | 来源 | 调整 |
| --- | --- | --- |
| `kgvae/vendor/wave_model.py` | `WAVE/wave/model.py` | 原文件直接复制；实际使用 `GeneVAE` 的 encoder、fc_mu、fc_logvar、reparameterize、decoder |
| `kgvae/vendor/projector.py` | `MAP/MAP-KG/model/model.py` 的 `ResidualProjector` | 原类直接提取，保留层名以严格加载权重 |
| `kgvae/vendor/molbart/STMencoder_ddp.py` | `MAP/MAP-KG/model/mega_molbart/STMencoder_ddp.py` | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/tokenizer.py` | MAP-KG 同名文件 | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/util.py` | MAP-KG 同名文件 | 直接复制 |
| `kgvae/vendor/molbart/decoder.py` | MAP-KG 同名文件 | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/megatron_bart.py` | MAP-KG 同名文件 | 仅替换包内导入路径；源文件本身已有移除 Megatron/Apex 依赖的 PyTorch shim |

新编写的适配逻辑：

- WAVE 默认 `latent_dim=128`，本项目明确设为 256；调用真实采样潜变量，而非 WAVE 融合分支使用的 encoder 中间输出。推理使用均值。
- WAVE 原训练脚本的损失类型被保留，但监督对象改为表达差值，补充基线重建监督及 Pearson 数值稳定项。
- MAP 的四层双向 token 融合通过 `nn.TransformerEncoderLayer` 实现，不照搬依赖特定 Transformers 私有 API 的 Llama 类，也不声称与 Llama 主干参数等价。
- 本地 MAP `pert.py` 接收 `concs` 但没有在 forward 中使用；本项目按架构文档补上 `log1p(dose_um / reference_um)` 乘法剂量条件。它是明确的适配设计，不是复制原代码中不存在的实现。
- 冻结 MAP-KG 的两条分支离线执行，基因经过预训练 `gene_projector`；缓存不是随机嵌入或未经知识投影的原始 ESM 特征。
- 未复用 WAVE 的 Morgan 指纹、表达空间融合 MLP、原始随机逐行划分逻辑，因为它们与目标架构/避免重复条件泄漏的要求不符。

预处理适配（多剂量 WAVE 流程）：

- 固定 24 h，保留所有有效正剂量；同药物/细胞系/剂量/时间的重复组成条件，至少两个重复。
- `kgvae/consensus.py` 根据 CMap 的 [modzs.m](https://github.com/cmap/cmapM/blob/master/sig_tools/math/modzs.m) 数学步骤独立实现 NumPy/SciPy 版本，并显式启用相关性与权重下界裁剪；没有复制该仓库源码，也不声称获得了 WAVE 原始预处理脚本。
- 在满足重复门槛的条件中按规范化 SMILES 去重统计化合物，每细胞系至少 50 种；DMSO 基线为同细胞系同时间所有对照的算术平均。
- MODZ 作用于 Level 3 的 978 个直接测量基因表达，不产生或读取 Level 5 Z-score。
