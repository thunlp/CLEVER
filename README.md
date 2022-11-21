# Visually Grounded Commonsense Knowledge Acquisition

The code and datasets of our AAAI 2023 paper [Visually Grounded Commonsense Knowledge Acquisition](https://not_available_yet).

## Overview

![CLEVER Framework](figs/framework.jpg)

In this work, we propose to formulate Commonsense Knowledge Extraction (CKE) as a distantly supervised multi-instance learning problem and present a dedicated CKE framework CLEVER that integrate VLP models with contrastive attention to deal with complex commonsense relation learning. You can find more details in our [paper](https://not_available_yet).


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Data Preparation

Check [DATASET.md](DATASET.md) for data preparation.

## Training

```sh
# Prepare dataset according to 'Data Preparation' Section

cd src/Oscar
bash train.sh
```



## Citation

Please consider citing this paper if you use the code:

```bib
@article{yao2023clever,
  title={Visually Grounded Commonsense Knowledge Acquisition},
  author={Yuan Yao, Tianyu Yu, Ao Zhang, Mengdi Li, Ruobing Xie, Cornelius Weber, Zhiyuan Liu, Hai-Tao Zheng, Stefan Wermter, Tat-Seng Chua, Maosong Sun},
  journal={Proceedings of AAAI},
  year={2022}
}
```

## License

CLEVER is released under the MIT license. See [LICENSE](LICENSE) for details.

## Acknowledge

Our implementation is based on the fantastic code of [Oscar](https://github.com/microsoft/Oscar).
