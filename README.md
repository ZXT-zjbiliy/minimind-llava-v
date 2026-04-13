# MiniMind LLaVA-V

* 本项目基于 `MiniMind` 语言模型与 `LLaVA-OneVision` 风格视觉分支，构建了一个轻量级多模态训练脚手架。
* 项目重点在于打通 `vision tower -> projector -> MiniMind decoder -> multimodal dataset -> training` 的最小闭环，用于多模态结构接入、训练流程验证与后续实验迭代。
* 当前版本已经具备视觉塔加载、projector 映射、多模态数据集处理、多模态预训练、全参 SFT、断点续训与 `torchrun` 多卡训练等能力。
* 项目定位为训练链路与结构实验仓库，当前未提供完整的多模态推理接口、对话 CLI 与生成封装。

---

# 📌 项目介绍

本项目来源于对 `MiniMind` 与 `LLaVA` 两条技术路线的本地化整合实践。其目标并非完整复刻原始多模态框架，而是在尽量保持工程简洁的前提下，将 MiniMind 语言模型与 LLaVA 风格视觉特征注入路径有效连接起来，形成可运行、可训练、可扩展的多模态实验基础设施。

相较于直接在大型多模态工程中改造，本项目更强调以下几点：

- 尽量复用 `MiniMind` 现有语言模型与训练代码；
- 采用可直接加载的 Hugging Face vision tower 目录，降低视觉侧接入成本；
- 将多模态相关改动集中在 `model/`、`dataset/` 与 `trainer/` 层，便于定位和扩展；
- 保持训练链路透明，方便继续补全推理、评测、权重衔接与更复杂的多模态能力。

当前仓库适用于以下场景：

- 为 MiniMind 增加图文多模态输入能力；
- 验证视觉编码器与语言模型 hidden states 的对接方式；
- 快速开展 projector 结构实验；
- 构建后续多模态 SFT、指令数据、推理接口与部署能力的基础代码框架。

---

#### 🎉 当前已实现内容

- `MiniMind` 语言模型主干接入；
- Hugging Face 格式 vision tower 加载；
- LLaVA 风格 projector 映射模块；
- 多模态数据集读取与 `collate_fn`；
- 多模态预训练脚本 `train_pretrain.py`；
- 多模态全参 SFT 脚本 `train_full_sft.py`；
- 单卡训练与 `torchrun` 多卡训练；
- checkpoint 保存、断点续训与可选 `torch.compile`。

#### 📎 当前未覆盖内容

- 独立的多模态推理 / 对话 CLI；
- `MiniMindLlavaVLM.generate()` 的完整实现；
- 明确区分 “基础 LLM 权重加载” 与 “VLM 权重继续微调” 的统一加载入口；
- 完整的线上推理服务封装。

---

# 📂 项目结构

```text
minimind_llava-v/
├─ model/
│  ├─ model_minimind.py
│  ├─ model_minimind_vlm.py
│  ├─ projector_minimind_llava.py
│  ├─ tokenizer.json
│  └─ tokenizer_config.json
├─ dataset/
│  ├─ lm_dataset.py
│  ├─ lm_dataset_vlm.py
│  ├─ pretrain_i2t.parquet
│  └─ sft_i2t.parquet
├─ trainer/
│  ├─ train_pretrain.py
│  ├─ train_full_sft.py
│  ├─ trainer_utils.py
│  └─ train_tokenizer.py
├─ weights/
│  └─ rice-vit-large-patch14-560/
├─ llava_reference/
├─ tools/
├─ out/
└─ checkpoints/
```

主要目录说明如下：

- `model/model_minimind.py`
  `MiniMind` 语言模型实现。
- `model/model_minimind_vlm.py`
  当前多模态模型主入口。负责视觉特征编码、projector 映射以及将图像 embedding 写入 `<|image_pad|>` 对应位置。
- `model/projector_minimind_llava.py`
  视觉 token 到 MiniMind hidden states 的映射模块，是后续结构实验的重点位置。
- `dataset/lm_dataset_vlm.py`
  多模态数据集实现，负责图像读取、占位符展开、对话模板构造与 batch 拼接。
- `trainer/train_pretrain.py`
  多模态预训练脚本，默认配置偏向“冻结 LLM 与 vision，仅训练 projector”。
- `trainer/train_full_sft.py`
  多模态全参 SFT 训练脚本，默认配置为“训练 LLM 与 projector，冻结 vision tower”。
- `trainer/trainer_utils.py`
  包含模型初始化、权重查找、vision tower 构建、checkpoint 保存与分布式训练工具。
- `llava_reference/`
  从 `LLaVA-OneVision-1.5` 拷贝的参考代码，用于结构对照，不作为当前仓库中的直接运行模块。

---

# ⚙️ 环境依赖

推荐环境如下：

- Python 3.10+
- PyTorch 2.x
- `transformers>=4.44`
- `numpy`
- `Pillow`
- `safetensors`
- `pyarrow`
- `datasets`
- `swanlab`（可选，用于训练日志记录）

安装示例：

```bash
pip install torch transformers safetensors numpy pillow
pip install pyarrow datasets swanlab
```

说明：

- 若仅使用 `json/jsonl` 数据格式，可不安装 `pyarrow`；
- 训练脚本参数中仍保留 `--use_wandb` 命名，但当前实际接入的是 `swanlab` 兼容接口。

---

# 📌 权重与路径约定

训练脚本默认采用以下目录约定：

- tokenizer：`./model`
- vision tower：`./weights/rice-vit-large-patch14-560`
- 预训练数据：`./dataset/pretrain_i2t.parquet`
- SFT 数据：`./dataset/sft_i2t.parquet`
- 导出权重：`./out`
- 断点续训：`./checkpoints`

其中，`--from_weight` 当前用于加载基础语言模型权重，默认会按以下顺序搜索：

1. 当前仓库的 `./out`
2. 当前仓库的 `./checkpoints`
3. 相邻目录 `../minimind/checkpoints`
4. 相邻目录 `../minimind/out`

若基础 `MiniMind` 预训练权重已存在于 `../minimind/out/pretrain_768.pth`，则当前仓库可直接复用。

使用时需注意：

- `hidden_size` 与 `num_hidden_layers` 必须与加载的 MiniMind 权重保持一致；
- 建议将多模态预训练产物命名为 `vlm_pretrain` 等独立名称，避免与基础语言模型 `pretrain` 权重混淆。

---

# 📌 快速开始

## 第 0 步

查看训练脚本参数：

```bash
python trainer/train_pretrain.py --help
python trainer/train_full_sft.py --help
```

如已准备好 `MiniMind` 基础权重，可直接开始多模态训练。

## 第 1 步：多模态预训练

默认策略如下：

- `freeze_llm=1`
- `freeze_vision=1`
- `freeze_projector=0`

即默认冻结语言模型与视觉塔，仅训练 projector。

示例命令：

```bash
python trainer/train_pretrain.py \
  --data_path ./dataset/pretrain_i2t.parquet \
  --from_weight pretrain \
  --save_weight vlm_pretrain \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 5e-4 \
  --freeze_llm 1 \
  --freeze_vision 1 \
  --freeze_projector 0
```

训练完成后，权重通常保存在：

- `out/vlm_pretrain_768.pth`
- `checkpoints/vlm_pretrain_768_resume.pth`

## 第 2 步：多模态全参 SFT

默认策略如下：

- `freeze_llm=0`
- `freeze_vision=1`
- `freeze_projector=0`

即默认训练语言模型与 projector，冻结 vision tower。

示例命令：

```bash
python trainer/train_full_sft.py \
  --data_path ./dataset/sft_i2t.parquet \
  --from_weight pretrain \
  --save_weight full_sft \
  --batch_size 4 \
  --accumulation_steps 1 \
  --learning_rate 1e-5 \
  --freeze_llm 0 \
  --freeze_vision 1 \
  --freeze_projector 0
```

## 第 3 步：多卡训练

当前脚本已兼容 `torchrun`，示例如下：

```bash
torchrun --nproc_per_node=4 trainer/train_pretrain.py \
  --data_path ./dataset/pretrain_i2t.parquet \
  --from_weight pretrain \
  --save_weight vlm_pretrain
```

## 第 4 步：断点续训

若需继续中断训练，可使用：

```bash
python trainer/train_pretrain.py --save_weight vlm_pretrain --from_resume 1
python trainer/train_full_sft.py --save_weight full_sft --from_resume 1
```

---

# 📌 数据格式说明

当前多模态数据集支持以下文件格式：

- `.parquet`
- `.json`
- `.jsonl`

常用字段包括：

- `conversations` 或 `messages`
- `images` 或 `image`
- `image_names`
- `image_bytes`

其中：

- `conversations/messages` 应为消息列表，每条消息至少包含 `role` 与 `content`；
- 图像既可由路径提供，也可由二进制字节流提供；
- 相对路径默认相对于数据文件所在目录解析，若传入 `--image_root`，则优先相对于 `image_root` 解析。

最小 `jsonl` 样例如下：

```json
{
  "conversations": [
    {"role": "user", "content": "<image>\n请描述这张图片。"},
    {"role": "assistant", "content": "这是一张……"}
  ],
  "images": ["demo.jpg"]
}
```

当前数据处理流程会自动完成以下步骤：

- 将 `<image>` 或 `<img>` 替换为完整视觉占位符序列；
- 使用 tokenizer 的 chat template 构造训练输入；
- 仅对 assistant 回复部分计算监督 loss；
- 将图像 resize / center crop 至 `560 x 560`。

在默认配置下：

- `patch_size = 14`
- `spatial_merge_size = 2`

因此，单张 `560 x 560` 图像对应：

- `40 x 40 = 1600` 个 patch token；
- 合并后得到 `400` 个 image token。

这部分 token 会与文本 token 共同占用上下文长度，多图场景下通常需要同步调大 `max_seq_len`。

如需快速查看 parquet 数据，可使用：

```bash
python tools/inspect_parquet.py ./dataset/pretrain_i2t.parquet --head 2
```

---

# 📌 常用参数说明

- `--vision_tower_path`
  指定视觉塔目录，默认值为 `./weights/rice-vit-large-patch14-560`。
- `--image_processor_config`
  指定图像预处理配置文件，默认使用视觉塔目录下的 `preprocessor_config.json`。
- `--from_weight`
  指定基础语言模型权重名，不带后缀，例如 `pretrain`。
- `--save_weight`
  指定当前阶段的保存名称，不带后缀，例如 `vlm_pretrain`、`full_sft`。
- `--freeze_llm / --freeze_vision / --freeze_projector`
  控制训练阶段中需要冻结的模块。
- `--max_seq_len`
  文本 token 与图像 token 共享的总上下文长度上限。
- `--use_compile`
  启用 `torch.compile`。
- `--use_wandb`
  当前实际对应 `swanlab` 兼容调用。

---

# 📌 注意事项

1. `llava_reference/` 中的代码主要用于结构参考，保留了原始工程的依赖方式，不应视为当前仓库中的直接运行模块。
2. 当前 `train_full_sft.py` 中的 `--from_weight` 主要用于加载基础 `MiniMind` 语言模型权重，并非显式意义上的 “加载完整 VLM checkpoint 后继续 SFT”。
3. 因此，若先在本仓库中得到 `out/pretrain_768.pth` 这类多模态权重，再直接以 `--from_weight pretrain` 启动 SFT，并不等价于严格意义上的 “VLM 预训练后继续微调”。
4. 更稳妥的实践方式是：
   - 基础阶段直接从 `../minimind` 的 LLM 权重启动；
   - 多模态预训练结果另存为 `vlm_pretrain`；
   - 后续如需串联 “VLM 预训练 -> VLM SFT”，建议补充单独的 VLM 权重加载逻辑。
5. 当前仓库重点在于训练链路验证与结构实验，不是完整推理项目。如需面向推理部署，仍需补充 `generate`、推理脚本、对话接口与服务封装。

---

# 📌 致谢

本项目的主要设计基础来自以下上游项目：

- [MiniMind](../minimind)
- [LLaVA-OneVision-1.5](../LLaVA-OneVision-1.5)

当前仓库在此基础上完成了本地化、轻量化的多模态接入与训练封装，用于后续继续开展结构实验、训练流程扩展与工程化完善。
