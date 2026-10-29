# 本地源码复用记录

原仓库保持不变。复制的许可文件位于 `kgvae/vendor/MAP_LICENSE`。K-Residual 不再包含 WAVE 源码，原先复制的 `wave_model.py` 与 `WAVE_LICENSE` 已删除。

| 本项目文件 | 来源 | 调整 |
| --- | --- | --- |
| `kgvae/vendor/projector.py` | `MAP/MAP-KG/model/model.py` 的 `ResidualProjector` | 原类直接提取，保留层名以严格加载权重；仅用于导出冻结 MAP-KG 特征 |
| `kgvae/vendor/molbart/STMencoder_ddp.py` | `MAP/MAP-KG/model/mega_molbart/STMencoder_ddp.py` | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/tokenizer.py` | MAP-KG 同名文件 | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/util.py` | MAP-KG 同名文件 | 直接复制 |
| `kgvae/vendor/molbart/decoder.py` | MAP-KG 同名文件 | 仅替换包内导入路径 |
| `kgvae/vendor/molbart/megatron_bart.py` | MAP-KG 同名文件 | 仅替换包内导入路径；源文件本身已有移除 Megatron/Apex 依赖的 PyTorch shim |

新编写的适配逻辑：

- 模型 `kgvae/model.py` 为新写的 K-Residual，不复制 MAP 或 WAVE 的预测网络。它借鉴了 MAP 的思路（知识嵌入的基因 token 与表达融合、剂量条件化的药物 token）和 WAVE 的思路（预测表达差值的残差输出）。
- 与 MAP 的差异：用纯交叉注意力代替 [药物, 细胞, 基因] 全序列自注意力；剂量通过 FiLM 注入，不再用标量乘法；没有 STATE 单细胞基础模型，细胞 token 由基线表达经 MLP 得到。
- 与 WAVE 的相同点：细胞只由基线表达表示，没有细胞系 ID。差异：去掉 β-VAE，改为不看药物的共性反应分支 b(x0) 加药物特异分支 g，g 的损失在残差上计算。
- 冻结 MAP-KG 的两条分支离线执行，基因经过预训练 `gene_projector`；缓存不是随机嵌入或未经知识投影的原始 ESM 特征。
- 未复用 WAVE 的 Morgan 指纹、β-VAE、表达空间融合 MLP 和原始随机逐行划分逻辑。

预处理适配（多剂量 WAVE 流程）：

- 固定 24 h，保留所有有效正剂量；同药物/细胞系/剂量/时间的重复组成条件，至少两个重复。
- `kgvae/consensus.py` 根据 CMap 的 [modzs.m](https://github.com/cmap/cmapM/blob/master/sig_tools/math/modzs.m) 数学步骤独立实现 NumPy/SciPy 版本，并显式启用相关性与权重下界裁剪；没有复制该仓库源码，也不声称获得了 WAVE 原始预处理脚本。
- 在满足重复门槛的条件中按规范化 SMILES 去重统计化合物，每细胞系至少 50 种；DMSO 基线为同细胞系同时间所有对照的算术平均。
- MODZ 作用于 Level 3 的 978 个直接测量基因表达，不产生或读取 Level 5 Z-score。
