<div align="center">
    <img src="images/dianjin_logo.png" alt="DianJin Logo" style="width: 200px;">
    <p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
        💜 <a href="https://tongyi.aliyun.com/dianjin" target="_blank" style="margin-left: 10px">Qwen DianJin Platform</a>  •
        🤗 <a href="https://huggingface.co/DianJin" target="_blank" style="margin-left: 10px">HuggingFace</a>  • 
        🤖 <a href="https://modelscope.cn/organization/tongyi_dianjin" target="_blank" style="margin-left: 10px">ModelScope</a> 
    </p>

**中文** | [**EN**](README.md)

</div>


## 🚀 最新动态
- **2025.10.11** "[FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol](./DianJin-TIR/technical%20report_FinMCP_Bench.pdf)" 由盈米基金联合合作伙伴发布，是首个基于MCP的、面向真实金融工具使用的LLM Agent基准数据集与评测体系。
- **2025.10.11** "[CARE: Cognitive-reasoning Augmented Reinforcement for Emotional Support Conversation](https://arxiv.org/abs/2510.05122)" 已发布！
- **2025.08.25** 🔥🔥🔥 "[Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models](https://arxiv.org/abs/2508.15202)" 已发布!
- **2025.08.18** 🔥🔥🔥《[DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model](https://www.arxiv.org/abs/2508.13238)》已发布并开源！
- **2025.08.08** 🔥🔥🔥《[Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](https://arxiv.org/abs/2508.04423)》已发布并开源！
- **2025.05.22** 《[M<sup>3</sup>FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](https://arxiv.org/abs/2506.02510)》已被 ACL-2025 正式录用！
- **2025.04.23** [DianJin-R1](DianJin-R1/README.md) 系列开源发布！此次发布包括 DianJin-R1-Data 数据集，以及两款强大的模型：DianJin-R1-7B 和 DianJin-R1-13B。查看我们的技术报告《[DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models](https://arxiv.org/abs/2504.15716)》，深入了解详情，并探索这些新模型的能力。
- **2025.01.06** [CFLUE](https://github.com/aliyun/cflue)数据集已经全部开源，现已开放下载！🚀🚀🚀
- **2024.05.16** 《[Benchmarking Large Language Models on CFLUE - A Chinese Financial Language Understanding Evaluation Dataset](https://arxiv.org/abs/2405.10542)》已被 ACL-2024 正式录用！ 🚀🚀🚀

目前已发布的**数据**和**模型**如下：

<table style="width: 100%; text-align: center;">
    <tr>
        <td></td>
        <th>ModelScope</th>
        <th>HuggingFace</th>
        <th>Paper</th>
    </tr>
    <tr>
        <th>Fin-PRM</th>
        <td><a href="https://modelscope.cn/organization/tongyi_dianjin">Fin-PRM</a></td>
        <td><a href="https://huggingface.co/DianJin">Fin-PRM</a></td>
        <td><a href="https://arxiv.org/abs/2508.15202">Paper</a></td>
    </tr>
    <tr>
        <th>DianJin-OCR-R1</th>
        <td><a href="https://modelscope.cn/organization/tongyi_dianjin">DianJin-OCR-R1</a></td>
        <td><a href="https://huggingface.co/DianJin">DianJin-OCR-R1</a></td>
        <td><a href="https://www.arxiv.org/abs/2508.13238">Paper</a></td>
    </tr>
    <tr>
        <th>CSC</th>
        <td><a href="https://www.modelscope.cn/datasets/tongyi_dianjin/DianJin-CSC-Data">CSC</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/DianJin-CSC-Data">CSC</a></td>
        <td><a href="https://arxiv.org/abs/2508.04423">Paper</a></td>
    </tr>
    <tr>
        <th>M<sup>3</sup>FinMeeting</th>
        <td colspan="2">Application Required</td>
        <td><a href="https://arxiv.org/abs/2506.02510">ACL-2025</a></td>
    </tr>
    <tr>
        <th rowspan="3">DianJin-R1</th>
        <td><a href="https://www.modelscope.cn/models/tongyi_dianjin/DianJin-R1-32B">DianJin-R1-32B</a></td>
        <td><a href="https://huggingface.co/DianJin/DianJin-R1-32B">DianJin-R1-32B</a></td>
        <td rowspan="3"><a href="https://arxiv.org/abs/2504.15716">Technical Report</a></td>
    </tr>
    <tr>
        <td><a href="https://www.modelscope.cn/models/tongyi_dianjin/DianJin-R1-7B">DianJin-R1-7B</a></td>
        <td><a href="https://huggingface.co/DianJin/DianJin-R1-7B">DianJin-R1-7B</a></td>
    </tr>
    <tr>
        <td><a href="https://www.modelscope.cn/datasets/tongyi_dianjin/DianJin-R1-Data">DianJin-R1-Data</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/DianJin-R1-Data">DianJin-R1-Data</a></td>
    </tr>
    <tr>
        <th>CFLUE</th>
        <td><a href="https://modelscope.cn/datasets/tongyi_dianjin/CFLUE">CFLUE</a></td>
        <td><a href="https://huggingface.co/datasets/DianJin/CFLUE">CFLUE</a></td>
        <td><a href="https://arxiv.org/abs/2405.10542">ACL-2024</a></td>
    </tr>
</table>


## 📝 简介
欢迎来到通义点金 👋

通义点金是阿里云打造的智能金融解决方案平台，专注于金融领域大型语言模型（LLM）和大型多模态模型（LMM）的研究与开发。我们为金融业务开发者提供一个创新且易于使用的人工智能应用开发环境。作为多种尖端人工智能技术的融汇平台，通义点金致力于推动智能金融领域的发展与创新。

在这里，您可以探索与体验最新的人工智能通用技术（AGI）应用，开启属于您的智能金融创新之旅！

欢迎您的莅临，共同探索智能金融的无限可能！

## ✨ 功能特色

### 💡 智能应用

提供标准的金融场景化API能力，如研报总结、资讯信息抽取、金融客服意图识别等 
- ✅ 金融服务: 如信用卡还款提醒、手机银行导航、续保提示、营销素材生成等
- ✅ 投研咨讯: 如研报总结、信息抽取、金融翻译、交易指标问答等
- ✅ 经营问数: 如经营指标问答、异常预警等智能化经营能力
- ...

### 💡 开放平台

为开发者提供一系列的金融接口和工具，更多能力建设，轻松完成对接和二次开发

- ✅ 文档问答: 定向优化文档解析和召回排序策略，提供更适合金融场景的知识库问答能力
- ✅ 指标问答: 具备指标问答，指标画图等能力，增加对金融专业知识的理解
- ✅ 多智能体: 包含多种类型节点的配置和编排，支持您在点金提供的能力基础上做更多个性化配置
- ...

## 🔖 引用

If you find our work helpful, feel free to give us a cite.

```
@article{care-esc,
      title={CARE: Cognitive-reasoning Augmented Reinforcement for Emotional Support Conversation}, 
      author={Jie Zhu and Yuanchen Zhou and Shuo Jiang and Junhui Li and Lifan Guo and Feng Chen and Chi Zhang and Fang Kong},
      journal={https://arxiv.org/abs/2510.05122},
      year={2025}
}

@article{fin-prm,
      title={Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models}, 
      author={Yuanchen Zhou and Shuo Jiang and Jie Zhu and Junhui Li and Lifan Guo and Feng Chen and Chi Zhang},
      journal={https://arxiv.org/abs/2508.15202},
      year={2025}
}

@article{dianjin-ocr-r1,
  title={DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model},
  author={Qian Chen, Xianyin Zhang, Lifan Guo, Feng Chen, Chi Zhang},
  journal={arXiv preprint arXiv:2508.13238},
  year={2025}
}

@article{csconv,
    title={Evaluating, Synthesizing, and Enhancing for Customer Support Conversation}, 
    author={Jie Zhu and Huaixia Dou and Junhui Li and Lifan Guo and Feng Chen and Chi Zhang and Fang Kong},
    journal = {https://arxiv.org/abs/2508.04423},
    year={2025}
}

@inproceedings{m3finmeeting,
    title = "M$^3$FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset",
    author = "Jie Zhu, Junhui Li, Yalong Wen, Xiandong Li, Lifan Guo, Feng Chen",
    booktitle = "Findings of ACL",
    year = "2025",
    pages = "244--266"
}

@article{dianjin-r1,
    title = "DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models", 
    author = "Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang",
    journal = "arxiv.org/abs/2504.15716",
    year = "2025"
}

@inproceedings{cflue,
    title = "Benchmarking Large Language Models on CFLUE - A Chinese Financial Language Understanding Evaluation Dataset",
    author = "Jie Zhu, Junhui Li, Yalong Wen, Lifan Guo",
    booktitle = "Findings of ACL",
    year = "2024",
    pages = "5673--5693",
}
```

## 🤝 联系我们
非常感谢您对通义点金系列的关注！如果您有兴趣向我们的研究团队或产品团队留言，欢迎通过我们的官方邮箱或者扫码加入钉群与我们联系：CFLUE@alibabacloud.com。我们的团队将竭诚为您提供帮助和支持。

<img src="images/dianjin_dingding.png" alt="DianJin Logo" style="width: 200px;">

## ⚠️ 免责声明

对于DianJin开源模型和数据的使用，我们不承担任何法律责任。用户在使用DianJin模型或数据时，需自行评估并承担可能存在的风险，并始终保持谨慎态度。
我们建议用户对模型输出的结果进行独立验证与分析，结合自身的实际需求和具体场景做出理性判断。
通过开源数据和模型，我们旨在为学术研究和行业应用提供有价值的工具，助力人工智能技术在数据分析、金融创新及其它相关领域的发展。
我们鼓励用户充分发挥创造力，深入探索DianJin模型的潜力，拓展其应用场景，共同推动人工智能技术在各领域的进步与实践。