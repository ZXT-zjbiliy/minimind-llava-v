# MiniMind LLaVA-V

一个把 MiniMind 语言模型和 LLaVA 风格视觉分支接起来的轻量多模态实验仓库。

这个目录的目标不是完整复刻原始 LLaVA 工程，而是尽快把下面这条链路在本地跑通：

`vision tower -> projector -> MiniMind decoder -> multimodal dataset -> training`

这样做的好处是，我们可以在不改动原始 `minimind` 和 `LLaVA-OneVision-1.5` 仓库的前提下，单独验证多模态接入方案、训练流程和数据格式。

## 当前状态

已经接通的部分：

- MiniMind LLM 主干
- Hugging Face 格式 vision tower 加载
- LLaVA 风格 projector
- 多模态数据集读取与 `collate_fn`
- 预训练与全参 SFT 训练脚本
- 单卡 / `torchrun` 多卡训练
- checkpoint 保存、断点续训、可选 `torch.compile`

当前还没有单独补齐的部分：

- 独立的多模态推理 / 对话 CLI
- `MiniMindLlavaVLM.generate()` 的完整封装
- 从 “VLM 预训练权重” 无缝衔接到 “VLM SFT” 的显式加载入口

如果你的目标是继续开发训练链路，这个仓库已经够用；如果你要直接做稳定推理服务，还需要再补一层推理封装。

## 目录结构

```text
minimind_llava-v/
├─ model/
│  ├─ model_minimind.py              # MiniMind 语言模型
│  ├─ model_minimind_vlm.py          # 多模态包装入口
│  ├─ projector_minimind_llava.py    # 视觉特征到 LLM hidden states 的映射
│  ├─ tokenizer.json
│  └─ tokenizer_config.json
├─ dataset/
│  ├─ lm_dataset_vlm.py              # 多模态数据集与 collate
│  ├─ pretrain_i2t.parquet           # 预训练样例数据
│  └─ sft_i2t.parquet                # SFT 样例数据
├─ trainer/
│  ├─ train_pretrain.py              # 多模态预训练
│  ├─ train_full_sft.py              # 多模态全参 SFT
│  └─ trainer_utils.py               # 模型初始化、checkpoint、DDP 等工具
├─ weights/
│  └─ rice-vit-large-patch14-560/    # vision tower 默认路径
├─ llava_reference/                  # 从 LLaVA-OneVision-1.5 拷来的参考代码
├─ out/                              # 导出的模型权重
└─ checkpoints/                      # 断点续训 checkpoint
```

核心文件说明：

- `model/model_minimind_vlm.py`
  当前项目的主入口。负责把视觉特征映射后写入 `<|image_pad|>` 对应的位置，再送入 MiniMind backbone。
- `model/projector_minimind_llava.py`
  当前默认 projector 会把视觉 patch token 按 `spatial_merge_size=2` 做合并后，再映射到 MiniMind hidden size。
- `dataset/lm_dataset_vlm.py`
  支持 `.parquet`、`.json`、`.jsonl`，并自动把文本中的 `<image>` / `<img>` 展开为完整视觉占位序列。
- `llava_reference/`
  仅作为结构参考，不是当前仓库里的可直接运行模块。

## 环境依赖

推荐环境：

- Python 3.10+
- PyTorch 2.x
- `transformers>=4.44`
- `numpy`
- `Pillow`
- `safetensors`
- `pyarrow`
- `datasets`
- `swanlab`（可选，用于训练日志）

一个可用的安装示例：

```bash
pip install torch transformers safetensors numpy pillow
pip install pyarrow datasets swanlab
```

说明：

- 如果你只用 `json/jsonl` 数据，可以不装 `pyarrow`。
- `--use_wandb` 这个参数名沿用了旧接口，但仓库里实际接的是 `swanlab` 的兼容调用。

## 权重与路径约定

训练脚本默认使用下面这些路径：

- tokenizer：`./model`
- vision tower：`./weights/rice-vit-large-patch14-560`
- 预训练数据：`./dataset/pretrain_i2t.parquet`
- SFT 数据：`./dataset/sft_i2t.parquet`
- 导出权重：`./out`
- 断点续训：`./checkpoints`

`--from_weight` 当前用于加载 MiniMind 语言模型权重，默认会按下面顺序查找：

1. 当前仓库的 `./out`
2. 当前仓库的 `./checkpoints`
3. 兄弟目录 `../minimind/checkpoints`
4. 兄弟目录 `../minimind/out`

这意味着如果你的基础 LLM 预训练权重已经放在 `../minimind/out/pretrain_768.pth`，这里可以直接复用。

需要注意两点：

- `hidden_size`、`num_hidden_layers` 要和你加载的 MiniMind 权重一致。
- 建议把 VLM 预训练产物命名成 `vlm_pretrain` 之类的名字，不要继续叫 `pretrain`，避免和基础 LLM 权重混淆。

## 快速开始

先看脚本参数：

```bash
python trainer/train_pretrain.py --help
python trainer/train_full_sft.py --help
```

如果你已经在相邻目录准备好了 `minimind` 的基础权重，通常可以直接开始训练。

## 训练流程

### 1. 多模态预训练

默认配置更偏向 “先训 projector”：

- `freeze_llm=1`
- `freeze_vision=1`
- `freeze_projector=0`

示例：

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

训练完成后，权重默认保存在：

- `out/vlm_pretrain_768.pth`
- `checkpoints/vlm_pretrain_768_resume.pth`

### 2. 多模态全参 SFT

默认配置是：

- `freeze_llm=0`
- `freeze_vision=1`
- `freeze_projector=0`

示例：

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

### 3. 多卡训练

脚本已经兼容 `torchrun`，例如：

```bash
torchrun --nproc_per_node=4 trainer/train_pretrain.py \
  --data_path ./dataset/pretrain_i2t.parquet \
  --from_weight pretrain \
  --save_weight vlm_pretrain
```

### 4. 断点续训

只要 `save_weight` 保持一致，就可以恢复：

```bash
python trainer/train_pretrain.py --save_weight vlm_pretrain --from_resume 1
python trainer/train_full_sft.py --save_weight full_sft --from_resume 1
```

## 数据格式

当前数据集类支持三种格式：

- `.parquet`
- `.json`
- `.jsonl`

最常见的字段约定如下：

- `conversations` 或 `messages`
- `images` 或 `image`
- `image_names`
- `image_bytes`

其中：

- `conversations/messages` 需要是一个消息列表，每条消息至少包含 `role` 和 `content`
- 图片既可以直接给路径，也可以给字节流
- 相对路径默认相对于数据文件所在目录；如果传了 `--image_root`，会优先相对于 `image_root` 解析

一个最小 `jsonl` 示例：

```json
{
  "conversations": [
    {"role": "user", "content": "<image>\n请描述这张图片。"},
    {"role": "assistant", "content": "这是一张……"}
  ],
  "images": ["demo.jpg"]
}
```

数据管线会自动做这些事情：

- 把 `<image>` 或 `<img>` 替换成完整视觉占位符序列
- 使用 tokenizer 的 chat template 组织对话
- 只对 assistant 回复部分计算监督 loss
- 把图像 resize / center crop 到 `560 x 560`

按当前默认配置：

- `patch_size = 14`
- `spatial_merge_size = 2`

所以每张 `560 x 560` 图像会对应：

- `40 x 40 = 1600` 个 patch token
- 合并后得到 `400` 个 image token

这件事非常重要，因为它会直接占用文本上下文长度。多图场景下，`max_seq_len` 往往要一起调大。

可以用下面的工具先检查 parquet 数据：

```bash
python tools/inspect_parquet.py ./dataset/pretrain_i2t.parquet --head 2
```

## 常用参数

- `--vision_tower_path`
  指向视觉塔目录，默认是 `./weights/rice-vit-large-patch14-560`
- `--image_processor_config`
  图像预处理配置，默认使用视觉塔目录下的 `preprocessor_config.json`
- `--from_weight`
  基础 LLM 权重名，不带后缀，例如 `pretrain`
- `--save_weight`
  当前阶段保存名，不带后缀，例如 `vlm_pretrain`、`full_sft`
- `--freeze_llm / --freeze_vision / --freeze_projector`
  控制训练哪些模块
- `--max_seq_len`
  文本和图像 token 的总长度上限
- `--use_compile`
  启用 `torch.compile`
- `--use_wandb`
  实际接的是 `swanlab`

## 已知限制与注意事项

1. `llava_reference/` 里的代码主要用于结构参考，里面仍保留了原始工程的依赖风格，不是当前仓库里的直接运行入口。
2. 当前 `train_full_sft.py` 的 `--from_weight` 主要面向 “MiniMind LLM 权重” 加载，不是显式的 “加载一整套 VLM checkpoint 继续 SFT”。
3. 因此，如果你先在当前仓库跑出了 `out/pretrain_768.pth` 这类 VLM 权重，再直接拿 `--from_weight pretrain` 去跑 SFT，语义上并不等于“从 VLM 预训练继续微调”。
4. 更稳妥的做法是：
   - 基础训练直接从 `../minimind` 的 LLM 权重启动
   - VLM 预训练结果另存为 `vlm_pretrain`
   - 后续如果要串联 “VLM 预训练 -> VLM SFT”，再补一个显式的 VLM 权重加载逻辑
5. 当前仓库重点是训练链路验证，不是成品推理项目；如果你要做推理，还需要补 `generate` / 推理脚本 / 对话接口。

## 致谢

这个仓库的主体思路来自两个上游项目：

- [MiniMind](../minimind)
- [LLaVA-OneVision-1.5](../LLaVA-OneVision-1.5)

本仓库做的是一层本地化、轻量化的多模态接入与训练脚手架，方便继续往下做结构实验和工程迭代。
