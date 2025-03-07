# DETR for License Plate Detection

This repository contains the implementation of the **DEtection TRansformer (DETR)** for license plate detection using the **UC3M-LP dataset**. DETR is an end-to-end object detection framework that eliminates the need for complex hand-crafted components such as anchor generation, non-maximum suppression (NMS), and region proposal networks (RPN). Instead, it formulates object detection as a direct set prediction problem and utilizes a transformer-based architecture to achieve high accuracy while maintaining conceptual simplicity.

## Dataset
The **UC3M-LP dataset** is used for training and evaluation. It is the largest open-source dataset for **European license plate detection and recognition**, specifically focusing on **Spanish plates**. It consists of:
- **1,975 images** of **2,547 vehicles**
- **12,757 plate characters**
- Polygonal plate labels for **rectification** and **character bounding boxes** for OCR fine-tuning

More details about the dataset can be found [here](link).

## Training Details
The model was trained using a **distributed approach** on **four NVIDIA A100-SXM4-40GB GPUs** for enhanced computational efficiency. Training setup:
- **Base Learning Rate:** `2×10−4` (increased from `1×10−4`)
- **Pretrained Checkpoint:** `checkpoint0300.pth` (Epoch 300)
- **Total Training Epochs:** 450

## Data Augmentation
To improve the robustness of the model under **real-world conditions**, various data augmentation techniques were implemented in `transforms.py` and integrated into the training pipeline in `coco.py`:
- **Color Jitter:** Random brightness, contrast, saturation, and hue adjustments.
- **Random Affine Transformations:** Rotation, translation, scaling, and shearing.
- **Random Perspective Transformations:** Simulates different viewing angles.
- **Gaussian Blur:** Mimics motion blur or out-of-focus conditions.
- **Random Occlusion:** Simulates occlusions due to dirt, debris, or other objects.
- **Fog and Rain Transformations:** Simulates adverse weather conditions.

## Experimental Results
Evaluation metrics on the test set:
- **AP@0.50:0.95:** `0.465`
- **AP@0.50:** `0.819`
- **AP@0.75:** `0.476`
- **AP for Large Objects:** `0.477`
- **AP for Medium Objects:** `0.143`
- **AP for Small Objects:** `-1.000`

Average Recall (AR):
- **AR@1:** `0.464`
- **AR@10:** `0.556`
- **AR@100:** `0.556`
- **AR for Large Objects:** `0.568`
- **AR for Medium Objects:** `0.242`
- **AR for Small Objects:** `-1.000`

## Model Performance on Real-World Images
The final model was evaluated on **real-world images** under **challenging conditions**, including:
- **Low-light environments**
- **Adverse weather conditions (rain, fog)**
- **Motion blur and occlusions**

The model delivered **highly accurate and reliable** results, demonstrating strong generalization and robustness.

## Installation
To set up the project environment, install dependencies using:

```bash
pip install -r requirements.txt

## Citation
If you use this model, please cite the following:

### DETR Paper
```bibtex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

### UC3M-LP Dataset
The dataset is derived from the UC3M-LP dataset:
[UC3M-LP Dataset](https://lsi.uc3m.es/2023/12/22/uc3m-lp-a-new-open-source-dataset/)

```bibtex
@article{uc3m-lp,
  title={UC3M-LP: A New Open-Source Dataset for License Plate Detection and Recognition},
  author={UC3M-LP Team},
  journal={LSI UC3M},
  year={2023},
  url={https://lsi.uc3m.es/2023/12/22/uc3m-lp-a-new-open-source-dataset/}
}
```

## License
This project is licensed under the MIT License.
