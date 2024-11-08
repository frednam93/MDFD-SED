# Multi-Dilated Frequency Dynamic Convolution for Sound Event Detection
Includes following methods: Dilated Frequency Dynamic Convolution, Partial Frequency Dynamic Convolution, Partial Dilated Frequency Dynamic Convolution and Multi-Dilated Frequency Dynamic Convolution

Official implementation of <br>
 - **Diversifying and Expanding Frequency-Adaptive Convolution Kernels for Sound Event Detection** (Accepted to INTERSPEECH2024) <br>
by Hyeonuk Nam, Seong-Hu Kim, Deokki Min, Junhyeok Lee, Yong-Hwa Park <br>
[![arXiv](https://img.shields.io/badge/arXiv-2406.05341-brightgreen)](https://arxiv.org/abs/2406.05341)<br>
 - **Pushing the Limit of Sound Event Detection with Multi-Dilated Frequency Dynamic Convolution** <br>
by Hyeonuk Nam, Yong-Hwa Park <br>
[![arXiv](https://img.shields.io/badge/arXiv-2406.13312-brightgreen)](https://arxiv.org/abs/2406.13312)<br>
 - **Self Training and Ensembling Frequency Dependent Networks with Coarse Prediction Pooling and Sound Event Bounding Boxes** (DCASE2024 Challenge Task4 technical report, **2nd rank**) <br>
by Hyeonuk Nam, Deokki Min, Seungdeok Choi, Inhan Choi, Yong-Hwa Park <br>
[![arXiv](https://img.shields.io/badge/arXiv-2406.15725-brightgreen)](https://arxiv.org/abs/2406.15725)
[![DCASE](https://img.shields.io/badge/DCASE-techreport-orange)](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Nam_38_t4.pdf)<br>


## Requirements
Python version of 3.7.10 is used with following libraries
- pytorch==1.8.0
- pytorch-lightning==1.2.4
- pytorchaudio==0.8.0
- scipy==1.4.1
- pandas==1.1.3
- numpy==1.19.2
- sed_scores_eval==0.0.4
- sebbs==0.0.0

other requrements in [requirements.txt](./requirements.txt)


## Datasets
You can download datasets by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). You need DESED real datasets (weak/unlabeled in domain/validation/public eval) and DESED synthetic datasets (train/validation).


## Test with saved models
You can test saved models by running:
```shell
python main.py
```
this example tests the best MDFD-CRNN model with class-wise median filter on truePSDS1.

To test MDFD-CRNNs with cSEBBs, run

```shell
python main.py -c ./configs/config_MDFDbest_sebb.yaml
```
then run
```shell
python sebbeval.py
```

To test DFD-CRNNs, run

```shell
python main.py -c ./configs/config_DFDbest_psds1.yaml
```
or
```shell
python main.py -c ./configs/config_DFDbest_psds2.yaml
```

To test PFD-CRNNs, run

```shell
python main.py -c ./configs/config_PFDbest.yaml
```

## Training
To train the model, you have to chage configs/config_*.yaml/training/test_only as False, and run:
```shell
python main.py
```
Trained model will be saved in `exps` folder.

## Reference
- [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task) <br>
- [Sound event detection with FilterAugment](https://github.com/frednam93/FilterAugSED) <br>
- [Temporal Dynamic CNN for text-independent speaker verification](https://https://github.com/shkim816/temporal_dynamic_cnn)
- [Frequency Dynamic Convolution-Recurrent Neural Network (FDY-CRNN) for Sound Event Detection](https://github.com/frednam93/FDY-SED)
- [Frequency & Channel Attention for Computationally Efficient Sound Event Detection](https://github.com/frednam93/lightSED)
- [Sound Event Bounding Boxes](https://github.com/merlresearch/sebbs)

## Citation & Contact
If this repository helped your works, please cite papers below! 3rd paper is about data augmentation method called FilterAugment which is applied to this work.
```bib
@article{nam2024dcase,
      title={Self Training and Ensembling Frequency Dependent Networks with Coarse Prediction Pooling and Sound Event Bounding Boxes}, 
      author={Hyeonuk Nam and Deokki Min and Seungdeok Choi and Inhan Choi and Yong-Hwa Park},
      year={2024},
      journal={arXiv preprint arXiv:2406.15725},
}

@article{nam2024pushing,
      title={Pushing the Limit of Sound Event Detection with Multi-Dilated Frequency Dynamic Convolution}, 
      author={Hyeonuk Nam and Yong-Hwa Park},
      year={2024},
      journal={arXiv preprint arXiv:2406.13312},
}

@article{nam2024diversifying,
      title={Diversifying and Expanding Frequency-Adaptive Convolution Kernels for Sound Event Detection}, 
      author={Hyeonuk Nam and Seong-Hu Kim and Deokki Min and Junhyeok Lee and Yong-Hwa Park},
      year={2024},
      journal={arXiv preprint arXiv:2406.05341},
}

@inproceedings{Nam2023,
    author = "Nam, Hyeonuk and Kim, Seong-Hu and Min, Deokki and Park, Yong-Hwa",
    title = "Frequency \& Channel Attention for Computationally Efficient Sound Event Detection",
    booktitle = "Proceedings of the 8th Detection and Classification of Acoustic Scenes and Events 2023 Workshop (DCASE2023)",
    address = "Tampere, Finland",
    month = "September",
    year = "2023",
    pages = "136--140",
}

@inproceedings{nam22_interspeech,
      author={Hyeonuk Nam and Seong-Hu Kim and Byeong-Yun Ko and Yong-Hwa Park},
      title={{Frequency Dynamic Convolution: Frequency-Adaptive Pattern Recognition for Sound Event Detection}},
      year=2022,
      booktitle={Proc. Interspeech 2022},
      pages={2763--2767},
      doi={10.21437/Interspeech.2022-10127}
}

@INPROCEEDINGS{nam2021filteraugment,
    author={Nam, Hyeonuk and Kim, Seong-Hu and Park, Yong-Hwa},
    booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Filteraugment: An Acoustic Environmental Data Augmentation Method}, 
    year={2022},
    pages={4308-4312},
    doi={10.1109/ICASSP43922.2022.9747680}
}
```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.
