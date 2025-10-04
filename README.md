# Awesome RL for Agents [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of reinforcement learning (RL) for agents.

> This list collects papers, tools, and demos that demonstrate how reinforcement learning can be applied to train or tune LLM/MLLM agents, with a focus on research-driven, computer-using, and tool-integrated agent behaviors. It is not associated with any survey or review — just a personal, living collection of resources on RL for agents. I’ll keep updating it as long as I’m still working in this area.
---

## Table of Contents

- [📚 Papers & Research](#-papers--research)
- [🕹️ Benchmarks](#-benchmarks)
- [🧪 Demos & Projects](#-demos--projects)
- [🧰 Toolkits & Frameworks](#-toolkits--frameworks)
- [📄 Tutorials & Blog Posts](#-tutorials--blog-posts)
- [🔗 Related Awesome Lists](#-related-awesome-lists)
- [🤝 Contributing](#-contributing)

---

## 📚 Papers & Research
### Survey & Review
- A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges [[Preprint'25]](https://arxiv.org/abs/2508.05668) [[AwesomeList]](https://github.com/YunjiaXi/Awesome-Search-Agent-Papers)
- **Deep Research Agents**: A Systematic Examination And Roadmap [[Preprint'25]](https://arxiv.org/abs/2506.18096) [[AwesomeList]](https://github.com/ai-agents-2030/awesome-deep-research-agent)

### RL for Computer-using Agents
- **OPENCUA**: OpenFoundations for Computer-Use Agents [[Preprint'25]](https://arxiv.org/abs/2508.09123) [[Code]](https://github.com/xlang-ai/OpenCUA)
- **ARPO**: End-to-End Policy Optimization for GUI Agents with Experience Replay [[Preprint'25]](https://arxiv.org/abs/2505.16282) [[Code]](https://github.com/dvlab-research/ARPO)
- **InfiGUI-R1**: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners [[Preprint'25]](https://arxiv.org/abs/2504.14239) [[Code]](https://github.com/Reallm-Labs/InfiGUI-R1)
- **Cracking the Code of Action**: a Generative Approach to Affordances for Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2504.17282)
- **UI-R1**: Enhancing Action Prediction of GUI Agents by Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs//2503.21620) [[Code]](https://github.com/lll6gg/UI-R1)
- **Digi-Q**: Learning Q-Value Functions for Training Device-Control Agents [[Preprint'25]](https://arxiv.org/abs/2502.15760) [[Code]](https://github.com/DigiRL-agent/digiq)
- **AutoWebGLM**: A Large Language Model-based Web Navigating Agent [[KDD'24]](https://dl.acm.org/doi/10.1145/3637528.3671620) [[Preprint'24]](https://arxiv.org/abs/2404.03648) [[Code]](https://github.com/THUDM/AutoWebGLM)

### RL for Research Agents
- Tree Search for LLM Agent Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2509.21240)
- **Tongyi DeepResearch**: A New Era of Open-Source AI Researchers [[Blog]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/) [[Code]](https://github.com/Alibaba-NLP/DeepResearch)
- **SSRL**: Self-Search Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2508.10874) [[Code]](https://github.com/TsinghuaC3I/SSRL)
- Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL [[Preprint'25]](https://arxiv.org/abs/2508.07976v2) [[Code]](https://github.com/inclusionAI/ASearcher)
- **MiroMind Open Deep Research** [[Blog]](https://miromind.ai/blog/miromind-open-deep-research) [[Code]](https://github.com/MiroMindAI)
- **ARPO**: Agentic Reinforced Policy Optimization [[Preprint'25]](https://arxiv.org/abs/2507.19849) [[Code]](https://github.com/dongguanting/ARPO)
- **Cognitive Kernel-Pro**: A Framework for Deep Research Agents and Agent Foundation Models Training [[Preprint'25]](https://arxiv.org/abs/2508.00414) [[Code]](https://github.com/Tencent/CognitiveKernel-Pro)
- **WebShaper**: Towards Autonomous Information Seeking Agency [[Preprint'25]](https://arxiv.org/abs/2507.15061) [[Code]](https://github.com/Alibaba-NLP/WebAgent)
- **WebSailor**: Navigating Super-human Reasoning for Web Agent [[Preprint'25]](https://arxiv.org/abs/2507.02592) [[Code]](https://github.com/Alibaba-NLP/WebAgent)
- **MMSearch-R1**: Incentivizing LMMs to Search [[Preprint'25]](https://arxiv.org/abs/2506.20670) [[Code]](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- **Kimi-Researcher**: End-to-End RL Training for Emerging Agentic Capabilities [[Blog]](https://moonshotai.github.io/Kimi-Researcher/)
- **R-Search**: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2506.04185) [[Code]](https://github.com/QingFei1/R-Search)
- **R1-Searcher++**: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2505.17005) [[Code]](https://github.com/RUCAIBox/R1-Searcher-plus)
- **ZeroSearch**: Incentivize the Search Capability of LLMs without Searching [[Preprint'25]](https://arxiv.org/abs/2505.04588) [[Code]](https://github.com/Alibaba-nlp/ZeroSearch)
- **DeepResearcher**: Scaling Deep Research via Reinforcement Learning in Real-world Environments [[Preprint'25]](https://arxiv.org/abs/2504.03160) [[Code]](https://github.com/GAIR-NLP/DeepResearcher)
- **ReCall**: Learning to Reason with Tool Call for LLMs via Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2503.19470) [[Code]](https://github.com/Agent-RL/ReCall)
- **Search-R1**: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2503.09516) [[Code]](https://github.com/petergriffinjin/search-r1)
- **R1-Searcher**: Incentivizing the Search Capability in LLMs via Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2503.05592) [[Code]](https://github.com/RUCAIBox/R1-Searcher)
- **Agentic Reasoning**: Reasoning LLMs with Tools for the Deep Research [[Preprint'25]](https://arxiv.org/abs/2502.04644) [[Code]](https://github.com/theworldofagents/Agentic-Reasoning)

### RL for Tool-using Problem Solver
- **VerlTool**: Towards Holistic Agentic Reinforcement Learning with Tool Use [[Preprint'25]](https://arxiv.org/abs/2509.01055) [[Code]](https://github.com/TIGER-AI-Lab/verl-tool)
- **VisualToolAgent (VisTA)**: A Reinforcement Learning Framework for Visual Tool Selection [[Preprint'25]](https://arxiv.org/abs/2505.20289) [[Code]](https://github.com/OoDBag/VisTA)
- **OTC**: Optimal Tool Calls via Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2504.14870)
- **ToolRL**: Reward is All Tool Learning Needs [[Preprint'25]](https://arxiv.org/abs/2504.13958) [[Code]](https://github.com/qiancheng0/ToolRL)
- **ReTool**: Reinforcement Learning for Strategic Tool Use in LLMs [[Preprint'25]](https://arxiv.org/abs/2504.11536)
- **Agent models**: Internalizing Chain-of-Action Generation into Reasoning models [[Preprint'25]](https://arxiv.org/abs/2503.06580) [[Code]](https://github.com/ADaM-BJTU/AutoCoA)
- **TORL**: Scaling Tool-Integrated RL [[Preprint'25]](https://arxiv.org/pdf/2503.23383) [[Code]](https://github.com/GAIR-NLP/ToRL)

### RL for Agent Memory
- **MemAgent**: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent [[Preprint'25]](https://arxiv.org/abs/2507.02259) [[Code]](https://github.com/BytedTsinghua-SIA/MemAgent)
- **MEM1**: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents [[Preprint'25]](https://arxiv.org/abs/2506.15841)

### Reinforcement Learning Scaling
- Group Sequence Policy Optimization [[Preprint'25]](https://arxiv.org/abs/2507.18071)
- **Skywork R1V2**: Multimodal Hybrid Reinforcement Learning for Reasoning [[Preprint'25]](https://arxiv.org/abs/2504.16656) [[Model]](https://huggingface.co/Skywork/Skywork-R1V2-38B)
- A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce [[Preprint'25]](https://arxiv.org/abs/2504.11343)
- **o3 & o4-mini**: Introducing OpenAI o3 and o4-mini [[Blog]](https://openai.com/index/introducing-o3-and-o4-mini/)
- **Skywork-OR1 (Open Reasoner 1)** [[Blog]](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680) [[Code]](https://github.com/SkyworkAI/Skywork-OR1)
- **VAPO**: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks [[Preprint'25]](https://arxiv.org/abs/2504.05118)
- **DAPO**: An Open-Source LLM Reinforcement Learning System at Scale [[Preprint'25]](https://arxiv.org/abs/2503.14476v1) [[Code]](https://github.com/BytedTsinghua-SIA/DAPO)
- **LIMR**: Less is More for RL Scaling [[Preprint'25]](https://arxiv.org/abs/2502.11886) [[Code]](https://github.com/GAIR-NLP/LIMR)
- **DeepSeek-R1**: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Preprint'25]](https://arxiv.org/abs/2501.12948)
- **Kimi k1.5**: Scaling Reinforcement Learning with LLMs [[Preprint'25]](https://arxiv.org/abs/2501.12599)

### Others
- **UFO**: A Simple "Try Again" Can Elicit Multi-Turn LLM Reasoning [[Preprint'25]](https://arxiv.org/abs/2507.14295) [[Code]](https://github.com/lichengliu03/unary-feedback)
- Self-Challenging Language Model Agents [[Preprint'25]](https://arxiv.org/abs/2506.01716v1)
- **MPO**: Boosting LLM Agents with Meta Plan Optimization [[Preprint'25]](https://arxiv.org/abs/2503.02682) [[Code]](https://github.com/WeiminXiong/MPO)

## 🕹 Benchmarks
- **BrowseComp-Plus**: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent [[Preprint'25]](https://arxiv.org/abs/2508.06600v1) [[Huggingface]](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
- **xbench**: Tracking Agents Productivity Scaling With Profession-Aligned Real-World Evaluations [[Preprint'25]](https://arxiv.org/abs/2506.13651) [[Website]](https://xbench.org/)
- **BrowseComp-ZH**: Benchmarking the Web Browsing Ability of Large Language Models in Chinese [[Preprint'25]](https://arxiv.org/abs/2504.19314) [[Code]](https://github.com/PALIN2018/BrowseComp-ZH)
- **BrowseComp**: a benchmark for browsing agents [[Blog]](https://openai.com/index/browsecomp/) [[Paper]](https://cdn.openai.com/pdf/5e10f4ab-d6f7-442e-9508-59515c65e35d/browsecomp.pdf) [[Code]](https://github.com/openai/simple-evals)
- **Computer Agent Arena**: Compare & Test AI Agents on Crowdsourced Real-World Computer Use Tasks [[Platform]](https://arena.xlang.ai/) [[Code]](https://github.com/xlang-ai/computer-agent-arena)
- **ScreenSpot-Pro**: GUI Grounding for Professional High-Resolution Computer Use [[Paper]](https://likaixin2000.github.io/papers/ScreenSpot_Pro.pdf) [[Code]](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
- **OSWorld**: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments [[NeurIPS'24]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html) [[Code]](https://github.com/xlang-ai/OSWorld)
- **SeeClick**: Harnessing GUI Grounding for Advanced Visual GUI Agents [[ACL'24]](https://aclanthology.org/2024.acl-long.505.pdf) [[Code]](https://github.com/njucckevin/SeeClick)

## 🧪 Demos & Projects

### RL-based LLM agent tuning
- **SkyRL-v0**: Train Real-World Long-Horizon Agents via Reinforcement Learning [[Blog]](https://novasky-ai.notion.site/skyrl-v0) [[Code]](https://github.com/NovaSky-AI/SkyRL)
- **Agent-R1**: Training Powerful LLM Agents with End-to-End Reinforcement Learning [[Code]](https://github.com/0russwest0/Agent-R1)
- **VAGEN**: Training VLM Agents with Multi-Turn Reinforcement Learning [[Code]](https://github.com/RAGEN-AI/vagen)
- **OpenManus-RL** [[Code]](https://github.com/OpenManus/OpenManus-RL) & **OpenManus** [[Code]](https://github.com/mannaandpoem/OpenManus)
- **RAGEN**: Training Agents by Reinforcing Reasoning [[Code]](https://github.com/ZihanWang314/ragen)

### RL-based LLM tuning
- **Open-Reasoner-Zero**: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model [[Preprint'25]](https://arxiv.org/abs/2503.24290) [[Code]](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- **simple_GRPO** [[Code]](https://github.com/lsdefine/simple_GRPO)

### MCP Agents
- **mcp-agent** [[Code]](https://github.com/lastmile-ai/mcp-agent)
- **Agent2Agent (A2A) protocol** [[Code]](https://github.com/google/A2A)

## 🧰 Toolkits & Frameworks
- **ROLL**: Reinforcement Learning Optimization for Large-Scale Learning [[Code]](https://github.com/alibaba/ROLL)
- **verl**: Volcano Engine Reinforcement Learning for LLM [[Code]](https://github.com/volcengine/verl)

## 📄 Tutorials & Blog Posts
- **Introducing ChatGPT agent**: bridging research and action [[Blog]](https://openai.com/index/introducing-chatgpt-agent/)
- **Context Engineering** [[Github]](https://github.com/davidkimai/Context-Engineering)
- **The Second Half** [[Blog]](https://ysymyth.github.io/The-Second-Half/)

## 🔗 Related Awesome Lists
- **Awesome-Search-Agent-Papers** [[List]](https://github.com/YunjiaXi/Awesome-Search-Agent-Papers) - covering search agent papers
- **Awesome Deep Research Agent** [[List]](https://github.com/ai-agents-2030/awesome-deep-research-agent) - covering deep research agents and benchmark results
- **Awesome-Agent-RL** [[List]](https://github.com/0russwest0/Awesome-Agent-RL) - covering RL for research agents
- **awesome-ml-agents** [[List]](https://github.com/tokarev-i-v/awesome-llm-rl-agents) - covering rl and agents before 2023

## 🤝 Contributing

Contributions are warmly welcome!

If you know a paper, tool, environment, or demo relevant to **RL for Agents**, feel free to open a pull request.

### Guidelines:
- Make sure the resource is publicly accessible and active.
- Use the same format as existing entries: `- **Name**: Title [Paper](link) [Code](link) – short description (optional).`
- Add entries under the most appropriate section.
- Avoid duplicates or resources that are already well-covered elsewhere.

We aim to keep this list high-quality, practical, and focused. Thank you for helping improve it! ✨
