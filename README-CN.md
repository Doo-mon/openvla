# OpenVLA: An Open-Source Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Evaluating OpenVLA**](#evaluating-openvla) | [**Project Website**](https://openvla.github.io/)

<hr style="border: 2px solid gray;"></hr>

## 最近更新
- [2024-08-14] Added new section, [Evaluating OpenVLA](#evaluating-openvla), with instructions for running BridgeData V2 WidowX robot evals
- [2024-07-08] Added new sections: [Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora), [Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla)
- [2024-06-13] Initial release

<hr style="border: 2px solid gray;"></hr>


一个简单可拓展的代码库，用于训练和微调通用机器人操作的 VLA 模型：


- **不同数据集混合**: 支持任意 RLDS 格式的数据集, 包括来自 [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) 的任意混合 .
- **轻松拓展**: 在 PyTorch FSDP 和 Flash-Attention 的支持下, 可以快速高效地训练 1B - 34B 参数的模型，并且易于适应模型架构 .
- **原生微调支持**: 内置支持（附带示例）各种形式的微调（完整、部分、LoRA）.

建立在 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 之上



## 开始

为了简单使用 OpenVLA 模型进行推理，我们提供了一个轻量级接口，该接口利用 HuggingFace 模块 `transformers` 中的 AutoClasses，并具有最少的依赖性。


例如，要在 WidowX 机器人的 [BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) 中加载 `openvla-7b` 以进行零样本指令跟踪：



```python
# 安装最小的依赖 (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# 加载 Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# 获取输入图像和格式化 prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# 预测 Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 机器人具体执行的代码...
robot.act(action, ...)
```

我们还提供了用于针对新任务和实施例 **微调 OpenVLA 模型** 的示例脚本（ [./vla-scripts/finetune.py](./vla-scripts/finetune.py) ） ；此脚本支持不同的微调模式 - 包括 [HuggingFace 的 PEFT 库](https://huggingface.co/docs/peft/en/index) 支持的（量化）低秩自适应 (LoRA)。

对于部署，我们提供了一个轻量级脚本，用于通过 REST API 提供 OpenVLA 模型（ [./vla-scripts/deploy.py](./vla-scripts/deploy.py) ），提供了一种将 OpenVLA 模型集成到现有机器人控制堆栈中的简便方法，从而消除了对强大的设备计算能力的需求。



## 预训练的 VLAs

发布了两个经过训练的 OpenVLA 模型 （checkpoints, configs, and model cards ）[下载](https://huggingface.co/openvla)：
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b): 论文中所使用的模型，在 Prismatic `prism-dinosiglip-224px` VLM ( 基于 DINOv2 和 SigLIP 的融合视觉骨架，以及 Llama-2 大语言模型 ) 上训练。 在一个涵盖了97万个trajectories的大型混合数据集 Open X-Embodiment ( [混合细节见 - "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py) )
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01): 开发期间使用的早期模型 , trained from
  the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM). Trained on the same mixture
  of datasets as [Octo](https://github.com/octo-models/octo), but for significantly fewer GPU hours than our final model 
  ([mixture details - see "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).


**关于模型许可和商业用途的明确说明**: While all code in this repository is released under an MIT 
License, our pretrained models may inherit restrictions from the underlying base models we use. Specifically, both the
above models are derived from Llama-2, and as such are subject to the 
[Llama Community License](https://ai.meta.com/llama/license/).

---

## 安装

> **注意**: 这些安装说明适用于全面预训练（和分布式微调）；如果只想使用 OpenVLA 模型运行推理（或执行轻量级微调），请参阅上面的说明！

该仓库使用 Python 3.10 创建, 但应该向后兼容任意 Python >= 3.8 的版本. 需要
PyTorch 2.2.* -- 安装指南[在此](https://pytorch.org/get-started/locally/). 该仓库的最新版本在以下环境被开发和测试:
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

**[5/21/24] Note**: Following reported regressions and breaking changes in later versions of `transformers`, `timm`, and
`tokenizers` 我们明确固定了上述依赖包的版本号. 我们正在努力实施全面测试，并计划尽快放宽这些限制.

一旦 PyTorch 被正确安装, 您可以通过可编辑安装模式在本地安装此包( 或通过 
`pip install git+https://github.com/openvla/openvla`):

```bash
cd openvla
pip install -e .

# 训练额外需要安装 Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# 确认安装 Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

如果您在安装过程中遇到任何问题，请提交 GitHub Issue

**注意:** 在 `vla-scripts/` 中可参阅完整的 OpenVLA 模型的完整训练和验证脚本。 请注意 `scripts/` 是原始仓库 `prismatic-vlms` 保留下来的文件夹 , 支持训练和验证 VLMs；虽然您可以使用此存储库来训练 VLM 和 VLA，但是尝试使用 `scripts/generate.py` 去让 OpenVLA 模型生成语言将不起作用 （因为我们仅训练当前 OpenVLA 模型来生成动作，并且仅生成动作）。

## 利用 LoRA 微调 OpenVLA

在本节中，将讨论通过 `transformers` 库对 OpenVLA 模型使用低秩自适应 (LoRA) 进行微调，如果你没有能力完全微调 7B 模型，建议使用该方法。
主要代码在 `vla-scripts/finetune.py`。 

下面展示了一个示例，说明如何通过 LoRA 微调 OpenVLA 的 chekpoint
([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
。 这里我们使用一个具有 80GB VRAM 的 A100 GPU 显卡在 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 数据集上进行微调。 ( 你也可以通过改变 batch size 的值，从而使用一个更小的GPU，但是至少有 ~27 GB 大小的内存 )

首先，下载 BridgeData V2 数据集:

```bash
# 切换目录到数据集文件夹
cd <PATH TO BASE DATASETS DIR>

# 下载完整的数据集 (124 GB) # -c 表示中断之后恢复下载
wget -r -c -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 将文件夹改名为 `bridge_orig` (注意: 省略这一步可能会导致后面的错误)
mv bridge_dataset bridge_orig
```

现在，启动 LoRA 微调脚本，如下所示。 注意 `--batch_size==16` 和 `--grad_accumulation_steps==1` 需要 ~72 GB GPU 内存。
如果你的 GPU 比较小，应该减小 `--batch_size` 然后增加 `--grad_accumulation_steps` 以进行稳定的训练。如果你有多个 GPU 且希望通过 PyTorch Distributed Data Parallel (DDP) 进行训练，在下面的命令中的 `--nproc-per-node` 设置 GPU 的数量。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

注意: 如果在上面的命令中，设置了 `--image_aug==False` ，你会在训练的日志文件中观察到接近 100% 的 `action_accuracy` ，因为 [`openvla-7b`](https://huggingface.co/openvla/openvla-7b)  模型已经在一个包括 BridgeData V2 的数据集超集中预训练过了（没有增强）。

如果要对不同的数据集进行 LoRA 微调，你可以在此下载混合的数据集 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
（参照[这个脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh)了解如何从OXE下载数据集的示例）。或者，如果你有一个不属于 OXE 的自定义数据集，你可以选择 (a) 转换数据集成为 RLDS 格式，使得其能够兼容我们的微调脚本（具体见[这个仓库](https://github.com/kpertsch/rlds_dataset_builder)），或者 (b) 使用自定义的 PyTorch Dataset wrapper （可以参阅 `vla-scripts/finetune.py` 中的注释）。 推荐使用方式 (a)，因为 RLDS 数据集和 dataloader 经过了更多的实验测试验证。

对于方法 (a)，在你把自己的数据集转换成为 RLDS 格式之后，你需要使用我们的 data loader 注册它，通过注册一个数据集配置文件 [config](prismatic/vla/datasets/rlds/oxe/configs.py#L54) 和一个数据集转换函数 [function](prismatic/vla/datasets/rlds/oxe/transforms.py#L828)。

一旦你合并了新的数据集之后，就可以使用上述命令 `vla-scripts/finetune.py` 进行 LoRA 微调。

如果你在运行中出现任何问题，可以访问 [VLA Troubleshooting](#vla-troubleshooting) 或者在 [OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=) 中寻找相同的 issue（包括已关闭的）。如果你找不到相似的，请随时创建新的 issue。

## 全面微调 OpenVLA

在本节中，我们将讨论使用 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 训练脚本通过原生 PyTorch 完全分片数据并行 (FSDP) 对 OpenVLA（所有 75 亿个参数）进行<ins>全面微调</ins>。全面微调更高级/更复杂，仅当您拥有足够的计算能力（例如，8 个 A100 GPU 的完整节点）并且 LoRA 微调不足以满足您的用例（例如，如果微调分布与预训练分布有很大差异）时才建议使用。否则，我们建议您尝试通过 LoRA 进行参数高效的微调。


为了进行全面微调，您需要下载与 Prismatic VLMs 代码库兼容的[另外一个版本的 OpenVLA 模型 checkpoint（openvla-7b-prismatic）](https://huggingface.co/openvla/openvla-7b-prismatic)，我们在此基础上构建了 OpenVLA 模型。您可以使用以下 git 命令下载（或者，您可以通过 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 下载）：


```bash
# 更改当前目录为模型ckpt保存目录
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# 下载 checkpoint (30 GB) -- 可能需要一点时间
git clone git@hf.co:openvla/openvla-7b-prismatic

# 如果上述命令未下载完整检查点，则需要通过 git 大文件存储 (LFS) 手动获取
# 注意：您可能需要配置 SSH 密钥才能使其正常工作
cd openvla-7b-prismatic
git lfs fetch --all
```


我们展示了如何使用具有8个 GPU 的单节点在 BridgeData V2 数据集上全面微调 OpenVLA。如果您希望使用不同数量的 GPU（或节点），则可以修改 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中的 VLA 训练配置


下载 BridgeData V2 数据集:

```bash
# 切换目录
cd <PATH TO BASE DATASETS DIR>

# 下载数据集 (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 改名 `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

接下来，创建一个（以 
`hf_...` 开头的字符串） [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) ，然后复制到一个在仓库根目录的文件  `.hf_token` (`openvla/.hf_token`)

```bash
# Go to openvla root directory
cd openvla

# Copy HF token value into token file. Replace "hf_..." with your own token value!
# See: https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```


现在，启动训练脚本。如果你想要使用不同数量的结点或GPU，需要对配置文件进行修改
 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 然后对应修改下面命令的参数 `--nnodes` 和 `--nproc-per-node`

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH TO openvla/openvla-7b-prismatic CHECKPOINT FILE: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --run_id <OPTIONAL RUN ID FOR WANDB LOGGING> \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_interval <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> \
  --is_resume False
```

注意，`--is_resume` 参数在上面设置为 `False`，因为我们是从一个预训练的ckpt进行微调而不是从一个停止的训练恢复

如果您的训练运行暂停，并且您希望从最新检查点恢复，请将 `--pretrained_checkpoint` 更改为最新检查点路径，然后设置 `--is_resume==True` 并分别指定 `--resume_step` 和 `--resume_epoch`。例如，如果您希望从名为 `step-010000-epoch-20-loss=0.0160.pt` 的检查点恢复训练，则需要设置 `is_resume==True`、`resume_step==10000` 和 `resume_epoch==20`。

注意：如果您运行上面的 BridgeData V2 微调命令，您应该会在训练日志中观察到接近 100% 的 Action Token 准确率，因为
[`openvla-7b`](https://huggingface.co/openvla/openvla-7b) 模型已经在包含 BridgeData V2 的数据集超集上进行了预训练。

要对不同数据集上的 OpenVLA 进行全面微调，您可以从 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) 下载数据集（请参阅 [此脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh)，了解如何从 OXE 下载数据集的示例）

或者，如果您有一个不属于 OXE 的自定义数据集，您可以将数据集转换为 RLDS 格式，使之与我们的微调脚本兼容
（请参阅 [此 repo](https://github.com/kpertsch/rlds_dataset_builder)，了解相关说明）。下载/转换数据集后，您需要修改以下文件：

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py): 通过创建实验类来添加新的训练配置，然后在文件底部的 `VLARegistry` 中注册它。
  * 确保为你的微调实验创建了一个新的独一无二的 `vla_id`，并对应修改了配置参数 - 例如： `expected_world_size` （GPU数量），
  `per_device_batch_size` （每个GPU的batchsize）， `global_batch_size` （总batchsize），`shuffle_buffer_size` （每个 GPU 的随机缓冲区中的样本数量）等等。请参阅文件顶部的 `VLAConfig` 类下的注释，以了解每个变量的用途。
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py): 在 `OXE_NAMED_MIXTURES` 字典中为您的微调 mixture 定义一种**新的 mixture 数据**。
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py): 为你的微调数据集定义一个**新的数据集转换函数**，并将其添加到文件底部的
`OXE_STANDARDIZATION_TRANSFORMS` 注册表中。
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py): 向 `OXE_DATASET_CONFIGS` 字典中添加一个新配置，指定微调数据集的 **observation 和 action spaces**。

完成上述步骤后，您可以使用 `vla-scripts/train.py` 脚本开始全面微调。确保将 `--vla.type` 参数设置为您在 `prismatic/conf/vla.py` 中添加的新 `vla_id`。


When you are finished with fine-tuning, you will need to convert the final model checkpoint to a version that is
compatible with the Hugging Face `transformers` library. See the [Converting Prismatic Models to Hugging Face](#converting-prismatic-models-to-hugging-face) section for instructions.

If you run into any issues, please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the
[OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=) (including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

### Converting Prismatic Models to Hugging Face

If you have used the Prismatic VLMs codebase to train your model (e.g., if you did full fine-tuning of OpenVLA on a
new dataset), you will need to convert the final checkpoint to a version that is compatible with Hugging Face
`transformers` AutoClasses. We discuss how to do so in this section.

Let's say your training run directory is `PRISMATIC_RUN_DIR` (e.g., `prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`).
Inside this directory, there should be a directory called `checkpoints` which contains saved model checkpoints (e.g.,
`step-295000-epoch-40-loss=0.2200.pt`). The Prismatic-to-Hugging-Face conversion script
([convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)) expects a checkpoint file
named `latest-checkpoint.pt`. Therefore, you should first create a symbolic link called `latest-checkpoint.pt` that
points to the checkpoint file that you wish to convert:

```bash
# Go to your Prismatic training run's `checkpoints` directory
cd PRISMATIC_RUN_DIR/checkpoints

# Create symbolic link pointing to your checkpoint file
ln -s <YOUR CHECKPOINT FILENAME> latest-checkpoint.pt
```

Then, launch the conversion script to convert the checkpoint from the Prismatic VLMs format to the Hugging Face format:

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
```

The command above will save the HF-compatible checkpoint in `output_hf_model_local_path`. Now you can load the checkpoint
with HF AutoClasses as normal, as shown below. Note that there is an additional necessary step to register the OpenVLA model
to HF AutoClasses before loading it because you are loading a locally saved checkpoint rather than one that is pushed to the
HF Hub (see [here](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)
for details).

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("<PATH TO CONVERTED CHECKPOINT DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<PATH TO CONVERTED CHECKPOINT DIR>",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

## Training VLAs from Scratch

We provide full instructions and configurations for training VLA models on (arbitrary subsets of) the
[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). If you run in to any issues with 
the following, see [VLA Troubleshooting](#vla-troubleshooting) below (or file a GitHub Issue).

### VLA Pretraining Datasets

We download and preprocess individual datasets from Open X-Embodiment in [RLDS format](https://github.com/google-research/rlds) following 
[this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh). See 
[mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) for the full list of component datasets (and mixture 
weights) we use to train `openvla-7b`. 
- **Important**: For the BridgeData V2 component, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it under the subdirectory `bridge_orig/`. 
  Replace any reference to `bridge` in the OXE code with `bridge_orig`.

### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use 
[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA 
training configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can 
add your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training via `torchrun`:

```bash
# Train VLA on BridgeData V2 with the Prismatic DINO-SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---




## Evaluating OpenVLA

### BridgeData V2 WidowX Evaluations

#### Setup

Clone the [BridgeData V2 WidowX controller repo](https://github.com/rail-berkeley/bridge_data_robot) and install the `widowx_envs` package:

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
pip install -e widowx_envs
```

Additionally, install the [`edgeml`](https://github.com/youliangtan/edgeml) library:
```bash
git clone https://github.com/youliangtan/edgeml.git
cd edgeml
pip install -e .
```

Follow the instructions in the `bridge_data_robot` README to create the Bridge WidowX Docker container.

#### Launching BridgeData V2 Evaluations

There are multiple ways to run BridgeData V2 evaluations. We describe the server-client method below.

In one Terminal window (e.g., in tmux), start the WidowX Docker container:

```bash
cd bridge_data_robot
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

In a second Terminal window, run the WidowX robot server:

```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

In a third Terminal window, run the OpenVLA policy evaluation script:

```bash
cd openvla
python experiments/robot/bridge/run_bridgev2_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b
```

If you run into any problems with evaluations, please file a GitHub Issue.

---















## 代码仓库结构

存储库/项目文件树的高级概述：

+ `prismatic` - 源包；提供模型加载、训练、数据预处理等核心实用程序。
+ `vla-scripts/` - 用于训练、微调和部署 VLA 的核心脚本。
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---

#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
