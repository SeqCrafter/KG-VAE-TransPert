# KG-VAE-TransPert：从下载到训练

根据 `模型架构.md` 实现：冻结 MAP-KG 药物/基因编码器，联合训练 256 维 WAVE β-VAE 与四层双向 Transformer，预测潜空间位移并输出 978 维表达差值。

```text
z0 = VAE.encode(x0)
dz_i = Transformer(Cell(z0), Drug(SMILES_i) × log1p(dose_i / 1 μM), GeneTokens)
delta = Decoder(z0 + sum_i(dz_i)) - Decoder(z0)
expression = x0 + delta
```

**本文主流程使用你指定的 LINCS2020 Level 3 Control/Case 两个文件，不再使用 GSE92742 作为默认数据源。** 以下步骤在云端执行，命令均从项目根目录运行。按要求，本地只编写和修改代码，没有运行语法检查、导入检查、测试或训练，也没有下载大型数据和权重。

## 第 1 步：在云端准备项目和环境

将整个项目上传到云端，例如 `/workspace/KG-VAE-TransPert`，然后进入目录：

```bash
cd /workspace/KG-VAE-TransPert

# 使用 Python 3.10 或更新版本。
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先按云端 GPU/CUDA 环境安装对应的 PyTorch，若云端已有配置好的 PyTorch 环境，可以直接使用该环境。再安装项目依赖：

```bash
pip install -r requirements.txt
pip install -e . --no-deps
mkdir -p data/raw weights/mapkg
```

默认配置使用单 GPU 和 BF16。若 GPU 不支持 BF16，在第 7 步将 `amp` 改为 `fp16` 或 `off`。当前训练脚本不是 DDP 多卡脚本。

## 第 2 步：下载 LINCS2020 数据

你提供的三个地址作为主数据源，文件统一放在项目的 `data/raw/`。可以直接运行：

```bash
wget -c -P data/raw https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx
wget -c -P data/raw https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_cp_n1805898x12328.gctx
wget -c -P data/raw https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt
```

还需要同一发布版本的两份辅助元数据：`geneinfo_beta.txt` 提供 landmark 基因标记和 gene symbol，`compoundinfo_beta.txt` 提供 `pert_id` 对应的 SMILES。实例元数据本身不能替代这两份映射。

```bash
wget -c -P data/raw https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
wget -c -P data/raw https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
```

也可以用项目下载器一次下载上述五份文件，**与 wget 二选一即可**：

```bash
python -m kgvae.download cmap --release lincs2020 --output data/raw
```

这两个 GCTX 已是未压缩文件，不需要 `--decompress`。下载器使用 `.part` 临时文件和 HTTP Range 续传；不要将 wget 未完成的目标文件直接交给下载器当成完整文件。数据量大，请在云端预留原始文件、配对数组和训练 checkpoint 的空间。

## 第 3 步：手动放置 MAP-KG 文件

从你提供的 [Google Drive 文件夹](https://drive.google.com/drive/folders/1cV0ZTk92PguKS2nyii6dLV0IfqoSDHsQ) 手动下载，放到项目根目录下的 **`weights/mapkg/`**。

本机对应目录是 `/Users/panxiaoguang/Desktop/KG-VAE-TransPert/weights/mapkg/`；上传到云端后保持同样的项目内相对路径即可。

| 截图中的文件 | 放置位置 | 本项目用途 |
| --- | --- | --- |
| `mapkg_encoder_v3.pt` | `weights/mapkg/mapkg_encoder_v3.pt` | 默认作为 MAP-KG 编码器 checkpoint 输入 |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_E...` | `weights/mapkg/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | ESM2 基因序列嵌入字典；截图名称被截断，完整名称以下载文件为准 |
| `epoch_3.pt` | 如需保留，放在 `weights/mapkg/epoch_3.pt` | 当前训练流程不使用；仅凭截图不能判定其中模型结构，不能直接当作本模型的恢复 checkpoint |

若下载的 ESM2 文件名与表格不同，可以保持原名，在第 6 步通过 `--esm-dict` 传入实际路径。这里需要的是作者提供的**序列嵌入字典**，不是把原始 FASTA 当作 `.pt` 输入。

### 还需要分子词表 bart_vocab.txt

**本项目默认使用 MolecularSTM 原仓库的 `bart_vocab.txt`。这是根据 MAP 本地源码确定的上游依赖选择，不是自行重建或随意替换词表。** 截图未包含该文件，需要单独下载到 `weights/mapkg/bart_vocab.txt`。

源码依据如下，后续维护时可直接追溯：

| MAP 本地源码 | 依据 |
| --- | --- |
| `MAP/MAP/model/mega_molbart/STMencoder_ddp.py` | 作者的词表路径明确指向 `/mnt/petrelfs/fengjinghao/CRAFT/VirtualCell/ours_v1/drug/MoleculeSTM/MoleculeSTM/bart_vocab.txt` |
| `MAP/MAP-KG/model/mega_molbart/STMencoder_ddp.py` | MAP-KG 训练分支默认读取本地 `bart_vocab.txt` 和 `molecule_model.pth`，没有自动下载逻辑 |
| `MAP/MAP-KG/model/mega_molbart/util.py` | `DEFAULT_CHEM_TOKEN_START = 272`，与 MolecularSTM 词表布局对应 |
| `MAP/MAP-KG/model/mega_molbart/tokenizer.py` | 按文件中的 token 顺序建立 ID 映射；训练加载路径没有重新生成、扩展或重排词表 |

没有在所查的 MAP 训练和加载代码中发现定制词表的步骤。因此，作者直接使用上游词表是有较强源码支持的判断；尚未确认的只是作者当时使用的具体上游提交版本及文件校验值，不能据此声称已经证明字节级一致。这不影响将上游原始词表作为默认依赖。

下载来源：[MoleculeSTM 原仓库词表](https://github.com/chao1224/MoleculeSTM/blob/main/MoleculeSTM/bart_vocab.txt)。

```bash
wget -O weights/mapkg/bart_vocab.txt https://raw.githubusercontent.com/chao1224/MoleculeSTM/main/MoleculeSTM/bart_vocab.txt
```

预训练模型和词表必须使用相同的 token 顺序，不能用自己的 SMILES 重新生成词表，也不能因为另一份词表大小相同就认为可以互换。词表内容不会由模型权重自动恢复。下载后保留该文件；本项目特征导出会记录其 SHA256，便于后续追溯，但记录本地哈希本身不等于与作者原文件做过比对。如果将来作者提供明确的配套版本或校验值，应以作者信息为准。

### 为什么 MAP README 有重训练命令，却仍然需要准备外部文件？

MAP README 的重训练命令并不包含所有外部资源的准备步骤。`MAP/MAP-KG/train_resume.sh` 只是启动 `train_resume.py`；后者默认还从 `./checkpoints/kg_v3/best_model.pt` 恢复训练。只下载源码、安装 requirements 后直接执行，仍可能因词表、分子基础权重、ESM 字典、知识图谱数据或恢复 checkpoint 缺失而报错。这是复现说明中省略的前置条件，不是必须自行训练一个新词表的信号。

| 资源 | 原作者代码中的来源或使用方式 | 本项目冻结 MAP-KG 流程是否需要 |
| --- | --- | --- |
| `bart_vocab.txt` | 指向 MolecularSTM 项目内的词表文件 | **需要**，采用上述上游原文件 |
| `molecule_model.pth` | 分子编码器构造时读取的本地文件；下游代码路径为 `/mnt/petrelfs/fengjinghao/molecule_model.pth`，未注明具体下载版本 | **完整 MAP-KG checkpoint 包含分子主干时无需另下载**：导出器直接提取其中参数用于构造，再严格加载；不能用任意同名基础模型补缺失权重 |
| MAP-KG checkpoint | MAP README 指向作者 Google Drive | **需要**，使用作者发布的 `mapkg_encoder_v3.pt`，内部结构尚需你在云端实际加载 |
| ESM2 基因嵌入字典 | `MAP/MAP-KG/model/model.py` 从本地 `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` 读取 | **需要**，使用作者网盘中的字典，不自行换模型重算以声称等价 |
| BioBERT | `train_resume.py` 指定 `dmis-lab/biobert-v1.1`；`model/model.py` 通过 Hugging Face `from_pretrained` 加载模型和 tokenizer | **不需要**，我们只导出已经训练好的分子、基因分支，不重新进行文本关系知识预训练 |
| 知识图谱预训练数据 | MAP README 指向 [RainGate/MAP-KG](https://huggingface.co/datasets/RainGate/MAP-KG) | **不需要**，本项目不重新预训练 MAP-KG |

上游 [MoleculeSTM README](https://github.com/chao1224/MoleculeSTM#3-checkpoints) 提供 MegaMolBART 下载脚本和 MolecularSTM checkpoint 来源，但 MAP 的本地路径没有明确说明其 `molecule_model.pth` 对应上游的哪个具体版本。不要把 MegaMolBART 基础权重、MolecularSTM 权重与知识预训练后的 MAP-KG 权重当成可互换文件。当前采用完整作者 MAP-KG checkpoint，正是为了沿用其中实际使用的分子参数。

**一致性原则：沿用作者 MAP-KG 权重、作者 ESM2 字典、MolecularSTM 原始词表，以及本地 MAP 的 tokenizer、分子编码与池化实现。** 当前导出器对缺失分支和维度不匹配采用严格报错，不随机补参数；非法或超长 SMILES 会提前拒绝，不沿用原编码器的甲烷替代或静默截断。保留在代码中的 `prepare_esm.py` 是早期实验性备选，不属于这里的作者一致性流程，也不应在正式流程中使用。

准备完成后的目录应为：

```text
KG-VAE-TransPert/
├── data/raw/
│   ├── level3_beta_ctl_n188708x12328.gctx
│   ├── level3_beta_trt_cp_n1805898x12328.gctx
│   ├── instinfo_beta.txt
│   ├── geneinfo_beta.txt
│   └── compoundinfo_beta.txt
├── weights/mapkg/
│   ├── mapkg_encoder_v3.pt
│   ├── Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
│   └── bart_vocab.txt
├── kgvae/
├── configs/default.yaml
└── scripts/train_cloud.sh
```

没有在本地打开你的 checkpoint，因此这里没有声称已确认内部参数。导出器要求 `smiles_encoder.*`、`smiles_projector.*` 和 `gene_projector.*`，支持裸 state_dict 及 `model_state_dict` / `state_dict` / `model` 容器，并去掉 DDP 的 `module.` 前缀。缺少必要分支时会报错，不用随机权重替代。通常分子主干已包含在完整 MAP-KG 权重中，无需额外准备 `molecule_model.pth`。

## 第 4 步：生成配对基线/扰动数据

```bash
python -m kgvae.preprocess cmap \
  --control-gctx data/raw/level3_beta_ctl_n188708x12328.gctx \
  --case-gctx data/raw/level3_beta_trt_cp_n1805898x12328.gctx \
  --inst-info data/raw/instinfo_beta.txt \
  --gene-info data/raw/geneinfo_beta.txt \
  --pert-info data/raw/compoundinfo_beta.txt \
  --min-replicates 2 --min-compounds 50 \
  --time-hours 24 \
  --output data/processed
```

预处理采用你提供的 WAVE/TranSiGen 流程，并将固定 10 μM 改为保留全部有效正剂量：

1. **固定时间为 24 h**，保留所有有效正剂量并按元数据单位转换为 μM。不传 `--dose-um`，不会限制为 10 μM，也不会使用 `nearest_dose` 重新分箱。
2. **只保留 978 个 landmark 基因**：从 `geneinfo_beta.txt` 的 `feature_space=landmark` 选择，Control/Case 分别按 gene ID 对齐。推断基因不进入 MODZ 相关性计算或模型。
3. **按扰动条件合并重复**：分组键为 `(cell_id, pert_id, dose_um, time_hours)`。同药物不同剂量独立分组；同条件跨板样本纳入同组，不再把板号作为条件或基线匹配键。剂量按单位转换后的数值精确分组，不自动合并近似剂量。
4. **至少 2 个重复**：`--min-replicates 2`，只有满足此门槛的条件才能产生训练签名。重复数按不同实例 sample_id 计数，元数据中的重复 sample_id 会报错。
5. **至少 50 种化合物/细胞系**：先排除无有效 SMILES、无 DMSO 基线和重复不足的条件，再统计每个细胞系跨全部剂量的不同规范化 SMILES 数量。`--min-compounds 50`，同化合物的多个剂量只计为一种，不按签名条数凑足 50。
6. **MODZ 共识**：对保留条件的重复样本计算加权表达，输出一条 978 维处理后表达。训练中的每一行代表一个条件，而非一个原始重复实例。
7. **细胞系 DMSO 基线**：对同一细胞系、同为 24 h 的 Level 3 DMSO vehicle 对照做算术平均，允许跨板。该细胞系所有药物/剂量共享此基线；不对对照使用 MODZ。

### MODZ 实现约定

对一个条件的重复表达矩阵（重复数 × 978）计算重复间 Spearman 相关系数，去掉对角线，将负相关截断为零。每个重复的未归一化权重为 `max(0.5 × 相关系数行和, 0.01)`，然后归一化至权重和为 1，用权重对**原始 Level 3 表达**加权。只有两个重复时权重相等；所有相关性为零时也等价于平均。常数向量与其他向量的相关性按零处理，非有限表达值直接报错。

此参数化参考 [CMap 官方 modzs.m](https://github.com/cmap/cmapM/blob/master/sig_tools/math/modzs.m) 的 `wt_avg`，并启用 `clip_low_cc=True, low_thresh_cc=0, clip_low_wt=True, low_thresh_wt=0.01`。你提供的段落没有说明具体裁剪参数，因此这里明确记录选择，不声称与未提供的 WAVE 预处理源码逐项一致。

**本项目使用 MODZ 权重聚合 Level 3 表达，不将其换成 Level 5 Z-score，也不额外 log1p。** 输出目标仍为共识处理表达减去 DMSO 均值。

`--qc-pass-only` 是可选的额外筛选：若启用，处理和对照实例都先要求 `qc_pass` 为真，再统计重复与化合物数量。默认不启用，因为你提供的流程没有给出该额外标准；也未自行添加相关性阈值或要求同一药物必须具备多个剂量。保留所有剂量并不保证每个药物都具有完整剂量梯度。

输出：

```text
data/processed/
  x0.npy                  # float32 [N,978]，细胞系 DMSO 均值
  xp.npy                  # float32 [N,978]，每个条件的 MODZ 共识表达
  metadata.csv            # 条件、剂量、replicate_count、replicate_ids、replicate_weights、control_count
  genes.csv               # gene_id,gene_symbol，与数组列顺序对应
  cell_filter_summary.csv # 满足重复门槛且有对照的细胞系，其独立化合物数和是否通过门槛
  control_samples.csv     # 实际使用的 DMSO 样本 ID、细胞系、时间
  provenance.json         # 筛选参数、各阶段数量、MODZ 参数及输入来源
```

这里是共识处理表达与细胞系对照均值的配对，不是真实配对单细胞。预处理分块读取 GCTX，MODZ 内存占用以单个条件的重复组为单位；训练使用 mmap。

**如果先前已经生成逐实例/同板配对数据，必须重新执行本步骤及第 5、6 步，并开始一个新的训练运行，不能继续使用旧 split 或旧 checkpoint 恢复。** 可将旧 `data/processed/`、`data/features/`、`runs/default/` 手动备份到其他目录，再按本文默认路径生成；程序不会覆盖已有配对数据。

可用 `--cells A375 MCF7` 选择细胞系，仍须满足化合物数量门槛。`--max-samples` 仅用于你自行做云端小规模流程检查，它在全部筛选完成后截取**共识条件数**，不截断重复组；截取后的调试子集不保证还包含每细胞系 50 种化合物，不能当作正式实验数据。

## 第 5 步：划分训练、验证、测试集

```bash
python -m kgvae.split \
  --root data/processed --method drug --seed 42 \
  --val-fraction 0.1 --test-fraction 0.1
```

生成 `data/processed/split_drug.json`。默认按规范化 SMILES 分组，约 80%/10%/10% **药物组**分到训练/验证/测试集，样本数比例不一定相同。验证和测试药物不出现在训练集，不把同药物不同重复随机分到不同集合。

也支持 `--method cell` 或 `--method condition`。使用其他方式时，记得将配置的 `data.split` 改为对应的 `split_cell.json` 或 `split_condition.json`。该训练协议使用单药样本，多药作为推理输入。

## 第 6 步：导出冻结 MAP-KG 嵌入

```bash
python -m kgvae.export_mapkg \
  --checkpoint weights/mapkg/mapkg_encoder_v3.pt \
  --vocab weights/mapkg/bart_vocab.txt \
  --esm-dict weights/mapkg/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --genes data/processed/genes.csv \
  --metadata data/processed/metadata.csv \
  --output data/features/mapkg.pt \
  --device cuda --batch-size 32
```

每个 SMILES 经过作者的冻结分子编码器和预训练投影器；每个原始 ESM2 基因向量经过预训练 `gene_projector`。结果按固定顺序缓存，后续训练不用每个 batch 重复运行这些冻结编码器。

ESM 字典需要包含全部 978 个基因。导出器按 NCBI Gene ID 和旧符号匹配内置的 10 个 LINCS 基因更名（例如 [HIST2H2BE → H2BC21](https://www.ncbi.nlm.nih.gov/gene/8349/)），保留输出的原始基因顺序和符号，并在 provenance 中记录实际 ESM 键。若字典使用其他键，可提供 `--gene-map mapping.csv`，两列为 `gene_symbol,esm_key`，显式映射优先。缺失基因、特征维度不符或 SMILES 超长均报错，不用零向量或其他分子替代。

词表缺失的 token 默认按原始预训练 tokenizer 映射为 UNK，并发出提示；可加 `--strict-vocab` 改为报错。加载作者 checkpoint 的 NumPy 标量元数据需要 PyTorch 2.6 及以上，仍使用受限的 `weights_only=True` 加载。

冻结编码器不使用训练集外的表达标签进行学习，因此此步骤可以缓存训练、验证、测试所有药物的 SMILES。

## 第 7 步：设置训练参数

编辑 `configs/default.yaml`。默认数据路径已经对应上述步骤：

```yaml
data:
  root: data/processed
  features: data/features/mapkg.pt
  split: data/processed/split_drug.json
  batch_size: 16
  num_workers: 4
```

在现有配置的 `train` 部分修改相应字段，不要用下面片段覆盖整个文件：

```yaml
train:
  device: cuda
  epochs: 200
  amp: bf16
  accumulation_steps: 1
  output: runs/default
```

CPU 使用 `device: cpu`、`amp: off`；不支持 BF16 的 GPU 改为 `fp16` 或 `off`。显存不足时减小 `data.batch_size`，增大 `train.accumulation_steps`，也可设置 `model.activation_checkpointing: true`。输入有 980 个 token，全注意力的显存开销不可忽略。

默认损失为：

```text
MSE(delta_hat, xp-x0)
+ 1.0 × [1 - mean(per_sample_Pearson(delta_hat, xp-x0))]
+ 0.01 × mean(sum_over_latent(KL))
+ 1.0 × MSE(Decoder(z0), x0)
```

损失权重位于 `loss.alpha/beta/gamma`。常数向量 Pearson 记为零以避免 NaN。没有额外的潜空间目标监督、方向损失或图损失。

## 第 8 步：开始联合训练

```bash
python -m kgvae.train --config configs/default.yaml
```

这一步从初始化开始联合训练 β-VAE、维度投影器和 Transformer，MAP-KG 始终冻结。训练时采样 z0，验证/推理时取均值 mu。两次解码始终使用同一个 z0。

产物保存在 `runs/default/`：

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 按验证集总损失选择的最佳模型 |
| `last.pt` | 最近完成 epoch 的状态，可恢复训练 |
| `config.yaml` | 本次配置 |
| `history.jsonl` | 每个 epoch 的训练/验证损失与指标 |
| `training_summary.json` | 最佳验证损失等信息 |

训练过程中不会使用测试集选择模型。不同实验应设置不同的 `train.output`，不要覆盖现有运行。

### 中断后继续训练

```bash
python -m kgvae.train \
  --config configs/default.yaml --resume runs/default/last.pt
```

恢复在 epoch 边界进行，恢复模型、优化器、学习率调度器、AMP scaler 和随机状态。可增加总 epoch 数；模型、数据、损失及种子配置应与原运行一致。**这里加载的是本项目生成的 last.pt，不是网盘里的 epoch_3.pt。**

## 第 9 步：手动评估测试集

训练完成后，由你运行：

```bash
python -m kgvae.predict \
  --checkpoint runs/default/best.pt \
  --features data/features/mapkg.pt \
  --device cuda --batch-size 16 \
  --output runs/test_predictions \
  evaluate \
  --root data/processed \
  --split data/processed/split_drug.json \
  --partition test
```

输出 `metrics.json`、`delta.npy`、`expression.npy`、`metadata.csv` 和 `genes.csv`。`delta.npy` 是预测变化量；`expression.npy=x0+delta`。输出目录已有预测时请使用新目录。

## 可选：准备好下载文件后，一条命令完成第 4–8 步

```bash
bash scripts/train_cloud.sh
```

脚本使用上文默认文件路径，不负责下载或安装。仅适合尚未生成 `data/processed` 的首次运行；已经完成预处理时，继续执行相应后续步骤即可，不应反复运行此脚本。

如果外部文件名不同，可显式指定：

```bash
MAP_KG_CKPT=/actual/path/mapkg_encoder_v3.pt \
MAP_VOCAB=/actual/path/bart_vocab.txt \
ESM_DICT=/actual/path/gene_embeddings.pt \
bash scripts/train_cloud.sh
```

## 可选：单药或多药预测，无需真实处理后表达

准备 `baseline.npy`（N×978，表达尺度与训练相同）及其列顺序对应的 `genes.csv`。请求 CSV 格式：

```csv
baseline_index,smiles,dose_um,time_hours
0,CCO,10,24
0,"[""CCO"",""CC(=O)O""]","[10,5]",24
```

SMILES 仅作格式示例，不代表有意义的药物实验。多药用 JSON 列表，每个药物对应一个 μM 剂量；不要按 `.` 或 `+` 拆 SMILES。时间必须与训练一致。

如果包含原缓存没有的新药，先按第 6 步重新导出缓存，保持相同 MAP-KG 权重、词表和 ESM 字典，并将两个参数改为：

```text
--metadata data/processed/metadata.csv requests.csv
--output data/features/mapkg_requests.pt
```

然后执行：

```bash
python -m kgvae.predict \
  --checkpoint runs/default/best.pt \
  --features data/features/mapkg_requests.pt \
  --output runs/combinations \
  requests --baseline baseline.npy \
  --genes data/processed/genes.csv --requests requests.csv
```

多药分别基于同一基线生成潜空间位移，再相加解码。零剂量药物被屏蔽，全零剂量时预测差值为零。该结构支持组合泛化，不保证准确刻画生物学协同作用。

## 可选：使用已有配对 h5ad 替代第 2、4 步

```bash
python -m kgvae.preprocess h5ad \
  --input /path/to/paired.h5ad --genes /path/to/landmark_genes.csv \
  --control-layer unpert_expr --perturbed-layer X \
  --smiles-column smiles --dose-column dose_um \
  --time-column time_hours --cell-column cell_id \
  --time-hours 24 --output data/processed
```

**此 h5ad 入口只导入已有配对数据，不重新执行 MODZ、重复数或细胞系化合物数筛选；输入需已按上述流程处理。**

基因 CSV 必须有 `gene_id,gene_symbol` 两列、978 行。默认 h5ad var_names 为 gene symbol；如果是 Entrez ID，加 `--gene-key gene_id`。ctrl/pert 格式使用 `--control-layer ctrl --perturbed-layer pert`。nM 剂量加 `--dose-factor 0.001`。缺少时间/剂量元数据时需按实验记录补齐，不会默认猜测。部分 anndata 版本会将 layers 载入内存，大规模数据优先使用 GCTX 入口。

## 源码复用与文件位置

- `kgvae/model.py`：模型；`kgvae/losses.py`：损失。
- `kgvae/download.py`、`kgvae/preprocess.py`、`kgvae/consensus.py`：下载、双 GCTX 筛选与 MODZ 共识。
- `kgvae/export_mapkg.py`：冻结编码器特征导出。
- `kgvae/split.py`、`kgvae/data.py`：划分、Dataset、DataLoader。
- `kgvae/train.py`、`kgvae/predict.py`：训练、恢复、评估和推理。
- `kgvae/vendor/`：直接复用的 MAP/WAVE 实现及原许可证。

复用和适配差异见 [THIRD_PARTY.md](THIRD_PARTY.md)。原始 `MAP/` 和 `WAVE/` 目录保持不变。旧 GSE92742 仍可通过 `download cmap --release gse92742` 及预处理 `--gctx` 单文件入口使用，但不要将不同版本的表达与元数据混用。
