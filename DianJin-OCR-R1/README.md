<div align="center">
    <h1><b>DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model</b></h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.04423-b31b1b.svg?logo=arXiv)](https://www.arxiv.org/pdf/2508.13238)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/aliyun/qwen-dianjin)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Dataset-orange.svg)](https://huggingface.co/DianJin)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-orange.svg)](https://modelscope.cn/organization/tongyi_dianjin)

[**中文**](README_zh.md) | **EN**

</div>

## Table of Contents
- [Introduction](#summary)
- [License](#license)
- [Citation](#cite)

## 📢 Introduction<a name="summary"></a>

Recent advances in large vision-language models (LVLMs) have enabled a new paradigm of end-to-end document image parsing, excelling in Optical Character Recognition (OCR) tasks such as text, table, and formula recognition. However, generative LVLMs, similarly to large language models (LLMs), are prone to hallucinations--generating words that do not exist in input images. Furthermore, LVLMs are designed for general purposes and tend to be less effective on OCR tasks compared to expert models that are trained on domain-specific datasets. In this paper, we propose DianJin-OCR-R1, a reasoning-enhanced framework designed to address these limitations through training reasoning-and-tool interleaved VLMs. Given a recognition instruction, our DianJin-OCR-R1 model first recognizes the content in the input image by its own OCR capabilities, and then calls other tools (i.e., other expert models) to obtain their results as references, finally looks again the image and rethinks about the reasoning process to provide the final recognized content. Since architectures of expert models are tailored for specific OCR tasks, which makes them less prone to hallucinations, their results can help VLMs mitigate hallucinations. Additionally, expert models are typically smaller in scale and easy to iterate, enabling performance improvements for VLMs at a lower cost. We evaluate our model on ReST and OmniDocBench, and experimental results show that our DianJin-OCR-R1 models consistently outperform their non-reasoning counterparts and expert OCR models, which proves the effectiveness of our method.

[Quick Start](src/quick_start.md)

## 📋 License<a name="license"></a>
![](https://img.shields.io/badge/License-MIT-blue.svg#id=wZ1Hr&originHeight=20&originWidth=82&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
This project adheres to [MIT License](https://lbesson.mit-license.org/).

## 🔖 Citation<a name="cite"></a>

If you use our dataset, please cite our paper.

```
@article{dianjin-ocr-r1,
  title={DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model},
  author={Qian Chen, Xianyin Zhang, Lifan Guo, Feng Chen, Chi Zhang},
  journal={arXiv preprint arXiv:2508.13238},
  year={2025}
}
```
