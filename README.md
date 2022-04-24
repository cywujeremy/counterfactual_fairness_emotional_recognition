# **Counterfactual Fairness in Speech Emotion Recognition**
This is the repository for the final group project of 11-785 Introduction to Deep Learning (Spring, 2022) at CMU. The main idea of the project is to introduce counterfactual data to mitigate potential gender bias in speech emotion recognition models.

The baseline model architecture is based on the paper [3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition](https://ieeexplore.ieee.org/document/8421023) of Chen, et al. But the training pipeline and train-test splits are reconstructed to fit the need in our research.

## **How to run it**

Run the following command at the root directory of this repository:

```
$ python train.py
```

## **Repo Structure**
```
├── archive
├── checkpoint
├── data
│   ├── __init__.py
│   ├── ExtractMel.py
│   ├── emotion_labels.py
│   ├── wav_file_reformat.sh
│   ├── IEMOCAP_eval.pkl
│   ├── IEMOCAP_test_data.npy
│   ├── IEMOCAP_test_gender.npy
│   ├── IEMOCAP_test_label.npy
│   ├── IEMOCAP_train_converted.pkl
│   ├── IEMOCAP_train.pkl
│   ├── IEMOCAP_valid_data.npy
│   ├── IEMOCAP_valid_gender.npy
│   └── IEMOCAP_valid_label.npy
├── log
├── nbs
│   ├── baseline_summary.ipynb
│   └── model_evaluation.ipynb
├── results
├── utils
│   ├── __init__.py
│   ├── fairness_eval.py
│   ├── training_tracker.py
│   └── zscore.py
├── datasets.py
├── model.py
├── train.py
└── README.md
```

* `model.py`: the implementation of the ACRNN model
* `datasets.py`: the dataloading logic
* `train.py`: the training pipeline