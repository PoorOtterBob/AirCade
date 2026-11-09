<div align="center">

# Spatiotemporal Causal Decoupling Model for Air Quality Forecasting

![Venue ICASSP%202025](https://img.shields.io/badge/Venue-ICASSP%202025-b8860b)
[![arXiv](https://img.shields.io/badge/arXiv-2505.20119-b31b1b.svg)](https://arxiv.org/abs/2505.20119)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c.svg)](https://pytorch.org/)

<p align="center">
  <a href="https://arxiv.org/abs/2505.20119">Paper</a> |
  <a href="#1-introduction-about-the-datasets">Datasets</a> |
  <a href="#3-model-running">Running</a> |
  <a href="#-citation">Citation</a> |
  <a href="#-license">License</a>
</p>

</div>

This is the official repository of our ICASSP 2025 paper. The proposed model, AirCade, addresses the limitations of current air quality forecasting models by integrating prior knowledge and current meteorological conditions into its architecture. It comprises a causal decoupling module designed to isolate the influence of prior knowledge on air quality index (AQI) data, and an environment simulation module that simulates the impact of future meteorological data on AQI. To account for the uncertainty of future meteorological conditions, AirCade implements perturbations and incorporates variance regularization and maximization in its optimization process. The model is evaluated on the largest available open-source PM2.5 dataset, demonstrating effective performance.

<br>

<img src='main.png' alt='Main figure of AirCade'>

<br>

## 1. Introduction about the datasets
### 1.1 Generating num 1 and num 2 sub-datasets from Knowair
In the experiments of our paper, we used num 1 and num 2 datasets with years from 2015 and 2017 respectively, which were followed by [Knowair](https://github.com/shuowang-ai/PM2.5-GNN). For example, you can download Knowair dataset from the provided [link](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) and please place the downloaded `Knowair.npy` file in the `AirCade_ICASSP` folder and complete the files in the `AirCade_ICASSP/data` folder.
  
```
python experiments/AirCade/train.py --device cuda:1 --num 1
```
Moreover, you can also generate the other years of data as the additional subdatasets. 

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [Knowair](https://github.com/shuowang-ai/PM2.5-GNN).

<br>

## 3. Model Running
To run AirCade on <b>Knowair</b>, for example, you may execute this command in the terminal:
```
bash run.sh
```
or directly execute the Python file in the terminal:
```
python experiments/AirCade/train.py --device cuda:1 --num 1
```

## 📄 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{ma2025spatiotemporal,
  title         = {Spatiotemporal Causal Decoupling Model for Air Quality Forecasting},
  author        = {Ma, Jiaming and Wang, Guanjun and Huang, Sheng and Yang, Kuo and Wang, Binwu and
                   Wang, Pengkun and Wang, Yang},
  booktitle     = {ICASSP 2025 -- 2025 IEEE International Conference on Acoustics, Speech and Signal Processing},
  year          = {2025},
  eprint        = {2505.20119},
  archivePrefix = {arXiv}
}
```

## 🙏 Acknowledgements

Built on [PM2.5-GNN (KnowAir)](https://github.com/shuowang-ai/PM2.5-GNN). The KnowAir data keeps the license of its original release.

## 📜 License

The code in this repository is released under the [MIT License](LICENSE).
