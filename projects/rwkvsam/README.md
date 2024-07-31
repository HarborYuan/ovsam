# RWKV-SAM

[Haobo Yuan<sup>1</sup>](https://yuanhaobo.me), 
[Xiangtai Li<sup>1,2</sup>](https://lxtgh.github.io),
[Lu Qi<sup>3</sup>](http://luqi.info),
[Tao Zhang<sup>2</sup>](https://scholar.google.com.hk/citations?user=3xu4a5oAAAAJ&hl=zh-CN),
[Ming-Hsuan Yang<sup>3</sup>](http://faculty.ucmerced.edu/mhyang/), 
[Shuicheng Yan<sup>2</sup>](https://yanshuicheng.info), 
[Chen Change Loy<sup>1</sup>](https://www.mmlab-ntu.com/person/ccloy/). 

[<sup>1</sup>S-Lab, Nanyang Technological University](https://www.mmlab-ntu.com/), 
[<sup>2</sup>Skywork AI](),
[<sup>3</sup>UC Merced](https://www.ucmerced.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2406.19369-b31b1b.svg)](https://arxiv.org/abs/2406.19369)

## 📰 News
* **` Jun. 28th, 2024`:** We release the arxiv paper and code for RWKV-SAM. The datasets, weights, and training scripts will be available soon. Please stay tuned.

## 👀 Overview
We introduce RWKV-SAM,  which includes an efficient segmentation backbone and a complete training pipeline to enable the high-quality segmentation capability for segment anything model.
Compared with the same-scale transformer model, RWKV-SAM achieves more than 2× speedup and can achieve better segmentation performance on various datasets.

<p>
  <img src="https://arxiv.org/html/2406.19369v1/x1.png" alt="RWKV-SAM overview">
</p>

## 📸 Demo
Our RWKV-SAM can achieve high-quality promptable segmentation with high efficiency at high resolution.
<p>
  <img src="https://arxiv.org/html/2406.19369v1/x6.png" alt="RWKV-SAM overview">
</p>

## 📚 Citation
```bibtex
@article{yuan2024rwkvsam,
    title={Mamba or RWKV: Exploring High-Quality and High-Efficiency Segment Anything Model},
    author={Yuan, Haobo and Li, Xiangtai and Qi, Lu and Zhang, Tao and Yang, Ming-Hsuan and Yan, Shuicheng and Loy, Chen Change},
    journal={arXiv preprint},
    year={2024}
}
```
