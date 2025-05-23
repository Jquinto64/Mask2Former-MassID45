# Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

[Bowen Cheng](https://bowenc0221.github.io/), [Ishan Misra](https://imisra.github.io/), [Alexander G. Schwing](https://alexander-schwing.de/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Rohit Girdhar](https://rohitgirdhar.github.io/)

[[`arXiv`](https://arxiv.org/abs/2112.01527)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Support major segmentation datasets: ADE20K, Cityscapes, COCO, Mapillary Vistas.

## Updates
* Add Google Colab demo.
* Video instance segmentation is now supported! Please check our [tech report](https://arxiv.org/abs/2112.10764) for more details.

## Installation

See [installation instructions](INSTALL.md).

## MassID45 Instructions
1. First download the MS-COCO pretrained R50 checkpoint using
```bash
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl
```

2. Make any desired modifications to the data augmentations with `mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py`, and modify the configs `configs/lifeplan/instance-segmentation/Base-Lifeplan-InstanceSegmentation.yaml` and `configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml` as you see fit. 

3. Modify the  `--dataset_path` and `MODEL.WEIGHTS` arguments in `train.sh` to reflect the locations of the MassID45 training data and pretrained model weights from Step 1. Then run the training script with 
```bash
sbatch train.sh
```
Outputs will be saved in the folder specified by `OUTPUT_DIR`.

4.  Replace `--dataset_img_path` and `--dataset_json_path` in `sahi_inference_256_datasets_v2.sh` with the locations of the MassID45 validation or testing data, respectively, then run inference with: 
```bash
sbatch sahi_inference_256_datasets_v2.sh
```
Results will appear in the `runs/predict` folder under the name specified by the `--exp_name` argument.

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

Replicate web demo and docker image is available here: [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).
