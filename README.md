# Open-Vocabulary SAM

[Haobo Yuan<sup>1</sup>](https://yuanhaobo.me), 
[Xiangtai Li<sup>1</sup>](https://lxtgh.github.io), 
[Chong Zhou<sup>1</sup>](https://chongzhou96.github.io), 
[Yining Li<sup>2</sup>](https://scholar.google.com/citations?user=y_cp1sUAAAAJ), 
[Kai Chen<sup>2</sup>](https://chenkai.site), 
[Chen Change Loy<sup>1</sup>](https://www.mmlab-ntu.com/person/ccloy/). 

[<sup>1</sup>S-Lab, Nanyang Technological University](https://www.mmlab-ntu.com/), 
[<sup>2</sup>Shanghai Artificial Intelligence Laboratory](https://www.shlab.org.cn/)

[[`Paper`](http://arxiv.org/abs/2401.02955)] 
[[`Project Page`](https://www.mmlab-ntu.com/project/ovsam)]
[[`Hugging Face Demo`](https://huggingface.co/spaces/HarborYuan/ovsam)]

## 👀 Overview
We introduce the Open-Vocabulary SAM, a SAM-inspired model designed for simultaneous interactive segmentation and recognition, leveraging two unique knowledge transfer modules: SAM2CLIP and CLIP2SAM. The former adapts SAM's knowledge into the CLIP via distillation and learnable transformer adapters, while the latter transfers CLIP knowledge into SAM, enhancing its recognition capabilities.

<p>
  <img src="https://www.mmlab-ntu.com/project/ovsam/img/ovsam_teaser.jpg" alt="OVSAM overview">
</p>

## 🔧Usage
To play with Open-Vocabulary SAM, you can:
1. Try the online demo on the [🤗Hugging Face Space](https://huggingface.co/spaces/HarborYuan/ovsam). Thanks for the generous support of the Hugging Face team.
2. Run the gradio demo locally by cloning and running the [repo](https://huggingface.co/spaces/HarborYuan/ovsam/tree/main) on 🤗Hugging Face:
    ```commandline
    git lfs install
    git clone https://huggingface.co/spaces/HarborYuan/ovsam ovsam_demo
    cd ovsam_demo
    conda create -n ovsam_demo python=3.10  && conda activate ovsam_demo
    python -m pip install gradio==4.7.1
    python -m pip install -r requirements.txt
    python main.py
    ```
3. Try to train or evaluate in this repo following the instructions below.

## ⚙️ Installation
We use conda to manage the environment.

Pytorch installation:
```commandline
conda install pytorch torchvision torchaudio cuda-toolkit pytorch-cuda==12.1 -c pytorch -c "nvidia/label/cuda-12.1.0"
```

mmengine installation:
```commandline
python -m pip install https://github.com/open-mmlab/mmengine/archive/refs/tags/v0.8.5.zip
```

mmcv installation (note that older version mmcv before this commit may cause bugs):
```commandline
TORCH_CUDA_ARCH_LIST="{COMCAP}" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10
```
Please ask ChatGPT to get `COMCAP`:
```text
What is the `Compute Capability` of NVIDIA {YOUR GPU MODEL}? Please only output the number, without text.
```

Other OpenMMLab packages:
```commandline
python -m pip install \
https://github.com/open-mmlab/mmdetection/archive/refs/tags/v3.1.0.zip \
https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v1.1.1.zip \
https://github.com/open-mmlab/mmpretrain/archive/refs/tags/v1.0.1.zip
```

Extra packages:
```commandline
python -m pip install git+https://github.com/cocodataset/panopticapi.git \
git+https://github.com/HarborYuan/lvis-api.git \
tqdm terminaltables pycocotools scipy tqdm ftfy regex timm scikit-image kornia
```

## 📈 Datasets
Datasets should be put in the `data/` folder of this project similar to [mmdet](https://mmdetection.readthedocs.io/en/latest/user_guides/tracking_dataset_prepare.html). Please prepare dataset in the following format.
### COCO dataset
```text
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   ├── train2017
│   ├── val2017
│   ├── panoptic_{train,val}2017/  # png annotations
```
### SAM dataset
```text
├── sam
│   ├── train.txt
│   ├── val.txt
│   ├── sa_000020
│   │   ├── sa_223750.jpg
│   │   ├── sa_223750.json
│   │   ├── ...
│   ├── ...
```
`train.txt` and `val.txt` should contain all the folders you need:
```text
sa_000020
sa_000021
...
```

## 🚀 Training
Please extract the language embeddings first.
```commandline
bash tools/dist.sh gen_cls seg/configs/ovsam/ovsam_coco_rn50x16_point.py 8
```

### SAM2CLIP
SAM feature extraction:
```commandline
bash tools/dist.sh test seg/configs/sam2clip/sam_vith_dump.py 8
```
SAM2CLIP training:
```commandline
bash tools/dist.sh train seg/configs/sam2clip/sam2clip_vith_rn50x16.py 8
```

### CLIP2SAM
CLIP2SAM training:
```commandline
bash tools/dist.sh train seg/configs/clip2sam/clip2sam_coco_rn50x16.py 8
```

## 🏃‍♀️Inference
```commandline
bash tools/dist.sh test seg/configs/ovsam/ovsam_coco_rn50x16_point.py 8
```
Please refer to [🤗Hugging Face](https://huggingface.co/HarborYuan/ovsam_models) to get the pre-trained weights:
```commandline
git clone https://huggingface.co/HarborYuan/ovsam_models models
```

## 📚 Citation
```bibtex
@article{yuan2024ovsam,
    title={Open-Vocabulary SAM: Segment and Recognize Twenty-thousand Classes Interactively},
    author={Yuan, Haobo and Li, Xiangtai and Zhou, Chong and Li, Yining and Chen, Kai and Loy, Chen Change},
    journal={arXiv preprint},
    year={2024}
}
```
## License <a name="license"></a>

This project is licensed under <a rel="license" href="https://github.com/HarborYuan/ovsam/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.
