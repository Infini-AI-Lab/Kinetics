# Kinetics: Rethinking Test-Time Scaling Laws
[Paper](https://arxiv.org/abs/2408.11049) | [Blog](https://infini-ai-lab.github.io/Kinetics-PR/)
## TL;DR
Kinetics scaling law challenges the existing Test-time Scaling laws by adopting a practical efficiency perspective. Kinetics also promotes sparse attention to establish a better trade-off between accuracy and inference-scaling cost.

## Abstract
<table>
  <tr>
    <td><img src="static/images/F1A.png" alt="Kinetics Scaling law on AIME24 dataset with Qwen3 series" width="400"></td>
    <td><img src="static/images/F1B.png" alt="Kinetics Sparse Scaling law on AIME24 dataset with Qwen3 series" width="400"></td>
  </tr>
</table>

We rethink test-time scaling laws from a *practical efficiency* perspective, revealing that the effectiveness of smaller models is significantly overestimated. Prior work, grounded in compute-optimality, overlooks critical memory access bottlenecks introduced by inference-time strategies (e.g., Best-of-*N*, long CoTs). Our holistic analysis, spanning models from 0.6B to 32B parameters, reveals a new *Kinetics Scaling Law* that better guides resource allocation by incorporating both computation and memory access costs. *Kinetics Scaling Law* suggests that test-time compute is more effective when used on models above a threshold than smaller ones. A key reason is that in TTS, attention—rather than parameter count—emerges as the dominant cost factor. Motivated by this, we propose a new scaling paradigm centered on *sparse attention*, which lowers per-token cost and enables longer generations and more parallel samples within the same resource budget. Empirically, we show that sparse attention models consistently outperform dense counterparts, achieving over **60 points** gains in low-cost regimes and over **5 points** gains in high-cost regimes for problem-solving accuracy on *AIME*, encompassing evaluations on state-of-the-art MoEs. These results suggest that sparse attention is essential for realizing the full potential of test-time scaling because, unlike training—where parameter scaling saturates—test-time accuracy continues to improve through increased generation.

## News
- **[2024/08]** Our paper is released on [arXiv](https://arxiv.org/abs/2408.11049).
- **[2024/08]** Added theoretical cost model analysis and comparison with FLOPs-based analysis.
- **TODO**: Add SGLang integration.

## Citation
```bibtex
@misc{chen2024magicdecbreakinglatencythroughputtradeoff,
  title={Kinetics: Rethinking Test-Time Scaling Laws},
  author={Ranajoy Sadhukhan and Zhuoming Chen and Yang Zhou and Emma Strubell and Beidi Chen},
  year={2024},
  eprint={2408.11049},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2408.11049},
}
```