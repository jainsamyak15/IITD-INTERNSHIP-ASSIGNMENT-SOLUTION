# DINO Object Detection on IIT Delhi Pedestrian Dataset

This repository contains the implementation and results of fine-tuning the DINO (DETR with Improved DeNoising Anchor Boxes) object detection model on the IIT Delhi pedestrian dataset.

## Project Overview

- **Task**: Fine-tuning DINO for pedestrian detection
- **Dataset**: IIT Delhi campus pedestrian dataset (200 images)
- **Model**: DINO-4scale with ResNet-50 (R50) backbone

## Results

After fine-tuning, the model achieved the following performance on the validation set:

- **Average Precision (AP) @ IoU=0.50:0.95**: 0.502
- **Average Precision (AP) @ IoU=0.50**: 0.854
- **Average Precision (AP) @ IoU=0.75**: 0.541

Performance across different scales:

- Small objects: AP = 0.412
- Medium objects: AP = 0.593
- Large objects: AP = 0.714

## Setup and Training

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Prepare the dataset according to the provided script
4. Fine-tune the model:

```bash
python main.py --output_dir ./output_iitd --with_box_refine --two_stage --num_classes 2 --epochs 5 --lr_drop 4 --coco_path /path/to/your/dataset --batch_size 2 --resume checkpoint0011_4scale.pth --config_file config/DINO/DINO_4scale_r50_iitd.py
```

## Dataset

The IIT Delhi campus pedestrian dataset can be downloaded from [this link](https://drive.google.com/drive/folders/11emGTWEdiKFwibpa9bPLreHEXYBnjZCx?usp=drive_link).


## Future Improvements

1. Increase the dataset size and diversity
2. Experiment with data augmentation techniques
3. Fine-tune hyperparameters, especially learning rate and number of epochs
4. Try different backbones or model architectures

## Acknowledgements

- [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)
- IIT Delhi for providing the pedestrian dataset
