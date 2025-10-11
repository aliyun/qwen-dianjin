<div align="center">
    <h1><b>DianJin-OCR-R1：通过推理与工具交替的视觉语言模型增强 OCR 能力</b></h1>


[![arXiv](https://img.shields.io/badge/arXiv-2508.04423-b31b1b.svg?logo=arXiv)](https://www.arxiv.org/pdf/2508.13238)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/aliyun/qwen-dianjin)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Dataset-orange.svg)](https://huggingface.co/DianJin)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-orange.svg)](https://modelscope.cn/organization/tongyi_dianjin)

**中文** | [**EN**](README.md)

</div>

## 目录
- [简介](#summary)
- [许可证](#license)
- [引用](#cite)

## 📢 简介<a name="summary"></a>

近期，大规模视觉语言模型（LVLM）的进展催生了一种端到端文档图像解析的新范式，在光学字符识别（OCR）任务（如文本、表格和公式识别）方面表现出色。然而，与大规模语言模型（LLM）类似，生成式 LVLM 容易出现幻觉——生成输入图像中不存在的单词。此外，LVLM 旨在通用，与在特定领域数据集上训练的专家模型相比，在 OCR 任务上的效果往往较差。在本文中，我们提出了 DianJin-OCR-R1，这是一种通过训练推理与工具交替的视觉语言模型来解决这些局限性的推理增强框架。给定识别指令，我们的 DianJin-OCR-R1 模型首先凭借自身的 OCR 能力识别输入图像中的内容，然后调用其他工具（即其他专家模型）获取其结果作为参考，最后再次查看图像并重新思考推理过程，以提供最终的识别内容。由于专家模型的架构是为特定的 OCR 任务量身定制的，这使得它们不太容易出现幻觉，因此它们的结果可以帮助视觉语言模型减少幻觉。此外，专家模型通常规模较小且易于迭代，从而能够以更低的成本提升视觉语言模型的性能。我们在 ReST 和 OmniDocBench 上对我们的模型进行了评估，实验结果表明，我们的 DianJin-OCR-R1 模型始终优于其非推理版本和专家 OCR 模型，这证明了我们方法的有效性。

[快速开始](src/quick_start.md)

## 📋 许可证<a name="license"></a>
![](https://img.shields.io/badge/License-MIT-blue.svg#id=wZ1Hr&originHeight=20&originWidth=82&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
本项目遵循 [MIT License](https://lbesson.mit-license.org/).

## 🔖 引用<a name="cite"></a>

如果您使用了我们的数据集，请引用我们的论文。

```
@article{dianjin-ocr-r1,
  title={DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model},
  author={Qian Chen, Xianyin Zhang, Lifan Guo, Feng Chen, Chi Zhang},
  journal={arXiv preprint arXiv:2508.13238},
  year={2025}
}
```
