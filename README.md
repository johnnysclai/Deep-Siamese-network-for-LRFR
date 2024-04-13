# Deep Siamese network for low-resolution face recognition

PyTorch implementation of [Deep Siamese network for low-resolution face recognition](https://ieeexplore.ieee.org/document/9689459).

### Citation
If you find this work useful for your research, please consider cite our paper:
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

### Requirements
- Python >= 3.8 ([Anaconda](https://www.anaconda.com) installation is recommended)
- Pytorch: https://pytorch.org/get-started/previous-versions/
- Other dependencies can be installed by running: `pip install -r ./Deep-Face-Recognition/src/requirements.txt`

### Dataset Preparation
* Datasets should be organized in the following form:
```markdown
# Download training datasets (where the faces are aligned) and organize it into the following form:
└──Projects
  ├── Deep-Face-Recognition
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

* Preprocess the training dataset, *VGGFace2* as mentioned in the paper, which align and resize the faces to 128x128 pixels:
```shell
cd Deep-Face-Recognition/tools/VGGFace2
python vggface2_resize.py # modify the paths in the script vggface2_resize.py
```
The `loose_landmark_train_test_remove_lfw_megaface.csv` is provided here: []


* The testing dataset, *SCface*, will be align and resize during evaluation. The landmarks are provided in the csv files. 

### Training
* Train the model with the following command. Modify arguments in `src/arguments/train_args.py` if necessary:
```shell
cd Deep-Face-Recognition/src
python train.py
```

### Evaluation
```shell
cd Deep-Face-Recognition/src
python evaluation.py # modify the paths in the script evaluation.py
```
