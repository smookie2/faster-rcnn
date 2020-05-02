# faster-rcnn
An implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

>[Paper](https://arxiv.org/abs/1506.01497)
>
>[Official Implementation in MATLAB](https://github.com/ShaoqingRen/faster_rcnn)

## Installation

Install system requirements

- Python 3.7

Install Python dependencies

```bash
pip install -r requirements.txt
```

## Dataset preparation

Dowload [CoCo dataset and annotation](http://cocodataset.org/#download), extract it, and put extracted files under directory `dataset` as follow:

```bash
faster-rcnn
├── dataset
│   ├── annotations
│   └── images
```