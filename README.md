# MarIS20K-QUT

Official repository for **MarIS20K-QUT**, a large-scale benchmark dataset for underwater dense image prediction.

This repository provides the dataset information and code for the paper:

**Frequency-Context Dynamic Convolution and a Large-Scale Benchmark Dataset for Underwater Dense Image Prediction**

## News

* **2026.XX.XX**: MarIS20K-QUT dataset and code are released.
* **2026.XX.XX**: Paper accepted by *Knowledge-Based Systems*.

## Introduction

MarIS20K-QUT is a large-scale underwater image dataset designed for dense image prediction tasks, including object detection and instance segmentation. The dataset contains diverse underwater scenes and object categories, providing a benchmark for evaluating underwater perception methods.

## Dataset

The MarIS20K-QUT dataset contains:

| Dataset      | Images | Categories | Annotation Type              |
| ------------ | -----: | ---------: | ---------------------------- |
| MarIS20K-QUT | 20,000 |         15 | Bounding box + instance mask |

Dataset download link:

[MarIS20K-QUT Dataset](YOUR_DATASET_LINK_HERE)

If the dataset link is not available yet, use:

```text
Coming soon.
```

## Repository Structure

```text
MarIS20K-QUT/
├── model/              # Model implementation
├── tools/              # Training and testing tools
├── setup.py            # Installation script
└── README.md           # Project documentation
```

## Installation

Clone this repository:

```bash
git clone https://github.com/xfwang-qut/MarIS20K-QUT.git
cd MarIS20K-QUT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided yet, please install the required packages according to your experimental environment.

## Usage

### Training

```bash
python tools/train.py
```

### Testing

```bash
python tools/test.py
```

Please modify the configuration files and dataset paths according to your local environment.

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{WANG2026116393,
  title = {Frequency-context dynamic convolution and a large-scale benchmark dataset for underwater dense image prediction},
  journal = {Knowledge-Based Systems},
  volume = {348},
  pages = {116393},
  year = {2026},
  issn = {0950-7051},
  doi = {https://doi.org/10.1016/j.knosys.2026.116393},
  author = {Xingfa Wang and Chenggang Dai and Chengjun Chen and Kunhua Liu and Mingxing Lin}
}
```

## Contact

For questions about the dataset or code, please contact:

* Xingfa Wang: [xfwang_qut@163.com](mailto:xfwang_qut@163.com)

## License

This project is released for academic research only. Please contact the authors for commercial use.

