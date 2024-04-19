# Deep Siamese network for low-resolution face recognition

The official PyTorch implementation of [Deep Siamese network for low-resolution face recognition](https://ieeexplore.ieee.org/document/9689459).

### Citation
If you find our work useful, please consider cite our paper:
```
@inproceedings{lai2021deep,
  title={Deep Siamese network for low-resolution face recognition},
  author={Lai, Shun-Cheung and Lam, Kin-Man},
  journal={IEEE International Conference on Image Processing},
  pages={1444-1449},
  year={2021},
  month={Dec.}
}
```

### Updates
- *2024/04/14*: The checkpoints and results of the original paper and reproduced by this repo are uploaded as v1.0.0 release.
- *2024/04/13*: The source code is first released.

### Requirements
- Python 3 ([Anaconda](https://www.anaconda.com) installation is strongly recommended)
- Install all Python dependencies by running: 
```
pip install -r ./Deep-Siamese-network-for-LRFR/src/requirements.txt
```

### Dataset Preparation
* Datasets should be organized in the following form:
```markdown
└──Projects
  ├── Deep-Siamese-network-for-LRFR
    ├── src
    │   ├── train.py
    │   ├── evaluation.py 
    │   ├── ...
    ├── data
    │   ├── scface_distance1.csv
    │   ├── scface_distance2.csv
    │   ├── scface_distance3.csv
    │   ├── scface_mugshot.csv
    ├── tools
    │   ├── VGGFace2
    │   │   ├── vggface2_resize.py
  ├── ...
├──Datasets
    ├── VGGFace2
    │   ├── train_test_128x128
    │   │   ├── ...
    │   ├── train
    │   │   ├── ...
    │   ├── test
    │   │   ├── ...
    │   ├── bb_landmark
    │   │   ├── loose_landmark_train_test_remove_lfw_megaface.csv
    ├── SCface
    │   ├── SCface_database
    │   │   ├── surveillance_cameras_all
    │   │   │   ├── ...
    │   │   ├── ...
    ├── QMUL-SurvFace
    │   ├── QMUL-SurvFace
    │   │   ├── Face_Verification_Test_Set
    │   │   │   ├── verification_images
    │   │   │   │   ├── ...
    │   │   │   ├── positive_pairs_names.mat
    │   │   │   ├── negative_pairs_names.mat
    │   │   ├── ...
    ├── LFW
    │   ├── lfw
    │   │   ├── ...
    ├── ...
```
- Datasets can be downloaded from the original sources:
  - [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
  - [LFW](http://vis-www.cs.umass.edu/lfw/)
  - [SCface](https://www.scface.org/)
  - [QMUL-SurvFace](https://qmul-survface.github.io/)

* Preprocess the training dataset, VGGFace2, which will align and resize the faces to 128x128 pixels:
```shell
cd Deep-Siamese-network-for-LRFR/tools/VGGFace2
python vggface2_resize.py # modify the paths in the script vggface2_resize.py
```
 * The `loose_landmark_train_test_remove_lfw_megaface.csv` is provided [here](https://github.com/johnnysclai/Deep-Siamese-network-for-LRFR/releases/download/v1.0.0/loose_landmark_train_test_remove_lfw_megaface.csv)

* The testing dataset, *SCface*, will be align and resize during evaluation. The landmarks are provided in the csv files in `data` folder. 

### Training
* Train the model with the following command. Modify arguments in `src/arguments/train_args.py` if necessary:
```shell
cd Deep-Siamese-network-for-LRFR/src
python train.py
```

### Evaluation
* Modify the paths of pretrained model in the script `evaluation.py` and run the following command to evaluate the model:
```shell
cd Deep-Siamese-network-for-LRFR/src
python evaluation.py
```

### Checkpoints and results
1. The original checkpoint and training log can be downloaded from [here](https://github.com/johnnysclai/Deep-Siamese-network-for-LRFR/releases/download/v1.0.0/original_paper_checkpoint.zip)
2. The reproduced checkpoint and training log can be downloaded from [here](https://github.com/johnnysclai/Deep-Siamese-network-for-LRFR/releases/download/v1.0.0/v1.0.0-reproduced_checkpoint.zip)

**Note**: 
- You may not obtain the same results as reported in the paper because the OS, hardware, and library version may vary.
- The training code and evaluation code in this repo is slightly different from the original code used in the paper, but the parameters setting are the same.
- The reproduced results are obtained by using the environment with Ubuntu 22.04.3 LTS, Python 3.10.12, PyTorch 2.1.2 with Cuda 12.1, and other libraries in `requirements.txt`.

LFW results (HR-to-LR setting):

|                                                    | 8 x 8  | 12 x 12 | 16 x 16 | 20 x 20 | 128x128 |
|----------------------------------------------------|--------|---------|---------|---------|---------|
| Reported in our paper                              | 94.8%  | 97.6%   | 98.2%   | 98.1%   | 99.1%   |
| Re-run the paper's checkpoint in above environment | 83.53% | 94.20%  | 97.23%  | 98.37%  | 99.08%  |
| Reproduced result                                  | 96.27% | 98.15%  | 98.67%  | 98.62%  | 98.67%  |

SCface results:

|                                                    | d1     | d2     | d3     |
|----------------------------------------------------|--------|--------|--------|
| Reported in our paper                              | 79.7%  | 95.7%  | 98.2%  |
| Re-run the paper's checkpoint in above environment | 78.92% | 96%    | 98.77% |
| Reproduced result                                  | 79.85% | 96.62% | 98.15% |

QMUL-SurvFace results:

| Method                                             | 30%    | 10%    | 1%     | 0.1%   | AUC    |
|----------------------------------------------------|--------|--------|--------|--------|--------|
| Reported in our paper                              | 75.09% | 52.74% | 21.41% | 11.02% | 80.03% |
| Re-run the paper's checkpoint in above environment | 75.15% | 52.41% | 21.86% | 10.49% | 80.06% |
| Reproduced checkpoint                              | 77.56% | 56.37% | 28.71% | 11.88% | 82.17% |

* The model in *our paper* and *reproduced checkpoint* are used a backbone of a 20-layers ResNet, same as [SphereFace paper](https://github.com/wy1iu/sphereface), which has approximately 28M parameters, and trained with VGGFace2 dataset.
