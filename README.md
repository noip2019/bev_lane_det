# BEV-LaneDet: a Simple and Effective 3D Lane Detection Baseline 
## Introduction
BEV-LaneDet is an efficient and robust monocular 3D lane detection system. First, we introduce the Virtual Camera, which unifies the intrinsic/extrinsic parameters of cameras mounted on different vehicles to ensure the consistency of the spatial relationship between cameras. It can effectively promote the learning process due to the unified visual space. Secondly, we propose a simple but efficient 3D lane representation called Key-Points Representation. This module is more suitable for representing the complicated and diverse 3D lane structures. Finally, we present a lightweight and chip-friendly spatial transformation module called Spatial Transformation Pyramid to transform multi-scale front view features into BEV features.  Experimental results demonstrate that our
work outperforms the state-of-the-art approaches in terms of F-Score, being 10.6% higher on the OpenLane dataset and 5.9% higher on the Apollo 3D synthetic dataset, with a speed of 185 FPS. Our paper has been accepted by cvpr2023 [arxiv](https://arxiv.org/abs/2210.06006).


- [Get Started](#getstart)
- [Benchmark](#benchmark)
- [Visualization](#visualization)


## <span id="getstart">Get Started</span>

### Installation
- To run our code, make sure you are using a machine with at least one GPU.
- Setup the enviroment 
```
pip install -r requirements.txt
```
### Training and evaluation on OpenLane
- Please refer to [OpenLane](https://github.com/OpenPerceptionX/OpenLane) for downloading OpenLane Dataset. For example: download OpenLane dataset to /dataset/openlane

- How to train:
    1. Please modify the configuration in the /tools/openlane_config.py
    2. Execute the following code:
```
cd tools
python3 train_openlane.py
```
- How to evaluation:
    1. Please modify the configuration in the /tools/val_openlane.py
    2. Execute the following code:
```
cd tools
python val_openlane.py
```

### Training and evaluation on Apollo 3D Lane Synthetic
- Please refer to [Apollo 3D Lane Synthetic](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset) for downloading Apollo 3D Lane Synthetic Dataset. For example: download OpenLane dataset to /dataset/apollo

- How to train:
    1. Please modify the configuration in the /tools/apollo_config.py
    2. Execute the following code:
```
cd tools
python3 train_apollo.py
```
- How to evaluation:
    1. Please modify the configuration in the /tools/val_apollo.py
    2. Execute the following code:
```
cd tools
python val_apollo.py
```

### Inference and evaluation on ONCE-3DLanes
- Please prepare the ONCE-3DLanes dataset under `data/ONCE-3DLanes`. The default config is [`tools/once_config.py`](./tools/once_config.py), and the training entry is:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 tools/train_once.py --config tools/once_config.py
```
- To test a trained checkpoint such as `ep0099.pth`, use the `bevlanedet` conda environment and run:
```
cd /home/lijishuo/bev_lane_det
bash test_once.sh /home/lijishuo/bev_lane_det/ep0099.pth
```
- The script [`test_once.sh`](./test_once.sh) will:
  1. run `tools/val_once.py` to regenerate prediction json files
  2. run `tools/eval_once_with_ratio.py --ratio-th 0.6` to report the ratio metric and ONCE official benchmark
- By default, predictions are saved to:
```
work_dirs/once_3dlanes/predictions/ep0099
```
- To choose specific GPUs, override `GPU_IDS`. For example:
```
cd /home/lijishuo/bev_lane_det
GPU_IDS=0,1,2,3 bash test_once.sh /home/lijishuo/bev_lane_det/ep0099.pth
```
- If you want to run the two stages manually, use:
```
cd /home/lijishuo/bev_lane_det
source /home/lijishuo/miniconda3/etc/profile.d/conda.sh
conda activate bevlanedet
CUDA_VISIBLE_DEVICES=0 python tools/val_once.py \
    --config tools/once_config.py \
    --checkpoint /home/lijishuo/bev_lane_det/ep0099.pth \
    --pred-root work_dirs/once_3dlanes/predictions/ep0099 \
    --skip-eval
python tools/eval_once_with_ratio.py \
    --pred-root work_dirs/once_3dlanes/predictions/ep0099 \
    --ratio-th 0.6
```

### Sequence-split training/testing
- A new split-based setup is available without changing the original labels. It uses sequence ids extracted from:
  - `BEVLaneDetCopy_toPKU/data/split_train_with_height_pitch_by_sequence_9.json`
  - `BEVLaneDetCopy_toPKU/data/split_test_with_height_pitch_by_sequence_1.json`
- Train with the new config and script:
```
bash train_once_sequence_split.sh
```
- Test with the new script:
```
bash test_once_sequence_split.sh /path/to/checkpoint.pth
```
- This split uses the same ONCE annotations, but filters samples by sequence. Training-time validation only runs `val_offical` with `ratio_th=0.6`; the ONCE official benchmark evaluation is skipped.
- You can still override dataset roots and image roots with `ONCE_3DLANES_ROOT`, `ONCE_3DLANES_TRAIN_IMAGE_ROOT`, and `ONCE_3DLANES_VAL_IMAGE_ROOT`.

## <span id="benchmark">Benchmark</span>

### Results of different models on OpenLane dataset

| Method | F-Score | X error  near | X error far | Z error near | Z error far|GFLOPs | TensorRT | PyTorch  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| Gen-LaneNet | 29.7 |0.309|0.877|0.16|0.75| 34 | – | 54FPS  |
| PersFormer  | 47.8 |0.322|0.778|0.213|0.681| 143 | – | 21FPS |
| Ours  | 58.4 | 0.309 |0.659|0.244|0.631| 53| 185FPS | 102FPS| 

### Results of different models on Apollo 3D Lane Synthetic (Balanced Scence)
| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 3D-LaneNet | 86.4 |0.068|0.477|0.015|0.202|
| Gen-LaneNet | 88.1 |0.061|0.486|0.012|0.214|
| CLGO | 91.9 |0.061|0.361|0.029|0.25|
| PersFormer  | 92.9 |0.054|0.356|0.01|0.234|
| Ours  | 98.7 | 0.016 |0.242|0.02|0.216|


### Virtual Camera

CPU implementation is here: [Virutal Camera on CPU](./csrc/README.md)

|  Hardware   | Single-thread  | Multi-thread |
|  ----  | ----  | ----|
| Apple M1  | 1.5ms | 0.5ms |
| Intel Xeon Platinum 8163 @ 2.5 GHz  |5.5ms  | 1.2ms|
| Nv V100| - | TODO |


## <span id="visualization">Visualization</span>
### OpenLane
Full-length (10 mins) video of OpenLane is here: [Video](./virtualization/ol.mp4) or you can find in https://www.youtube.com/watch?v=Mqh0N2cOctM

![OpenLane](./visualization/ol.gif)

### Apollo 3D Lane Synthetic
You can watch video of Apollo 3D Lane Synthetic in https://www.youtube.com/watch?v=WC36c4wO_QM
