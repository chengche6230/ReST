<h1 align="center">ReST 🛌 (ICCV2023)</h1>
<h3 align="center">ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking</h3>
<p align="center"><a href="https://github.com/chengche6230">Cheng-Che Cheng</a><sup>1</sup>, Min-Xuan Qiu<sup>1</sup>, <a href="https://www.cs.ccu.edu.tw/~ckchiang/">Chen-Kuo Chiang</a><sup>2</sup>, <a href="https://cv.cs.nthu.edu.tw/people.php">Shang-Hong Lai</a><sup>1</sup></p>
<p align="center"><sup>1</sup>National Tsing Hua University, Taiwan, <sup>2</sup>National Chung Cheng University, Taiwan</p>

## News
* 2023.8 Code release
* 2023.7 Our paper is accepted to [ICCV 2023](https://iccv2023.thecvf.com/)!

## Introduction
ReST, a novel reconfigurable graph model, that first associates all detected objects across cameras spatially before reconfiguring it into a temporal graph for Temporal Association. This two-stage association approach enables us to extract robust spatial and temporal-aware features and address the problem with fragmented tracklets. Furthermore, our model is designed for online tracking, making it suitable for real-world applications. Experimental results show that the proposed graph model is able to extract more discriminating features for object tracking, and our model achieves state-of-the-art performance on several public datasets.
<img src="https://github.com/chengche6230/ReST/blob/main/docs/method-overview.jpg" width="100%" height="100%"/>

## Requirements
### Installation
1. Clone the project and create virtual environment
    ```bash
    git clone https://github.com/chengche6230/ReST.git
   conda create --name ReST python=3.8
   conda activate ReST
    ```
2. Install (follow instructions):
   * [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
   * [DGL](https://www.dgl.ai/pages/start.html) (also check PyTorch/CUDA compatibility table below)
   * [warmup_scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr)
   * [py-motmetrics](https://github.com/cheind/py-motmetrics)
   * Reference commands:
     ```bash
     # torchreid
     git clone https://github.com/KaiyangZhou/deep-person-reid.git
     cd deep-person-reid/
     pip install -r requirements.txt
     conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
     python setup.py develop

     # other paskages (in /ReST)
     conda install -c dglteam/label/cu117 dgl
     pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
     pip install motmetrics
     ```

3. Install other requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Download pre-trained ReID model
   * [OSNet](https://drive.google.com/file/d/1z1l3FoEGfIon-JH1vEzUKSGfLmMnzFek/view?usp=sharing)

### Datasets
1. Place datasets in `./datasets/` as:
```text
./datasets/
├── CAMPUS/
│   ├── Garden1/
│   │   └── view-{}.txt
│   ├── Garden2/
│   │   └── view-HC{}.txt
│   ├── Parkinglot/
│   │   └── view-GL{}.txt
│   └── metainfo.json
├── PETS09/
│   ├── S2L1/
│   │   └── View_00{}.txt
│   └── metainfo.json
├── Wildtrack/
│   ├── sequence1/
│   │   └── src/
│   │       ├── annotations_positions/
│   │       └── Image_subsets/
│   └── metainfo.json
└── {DATASET_NAME}/ # for customized dataset
    ├── {SEQUENCE_NAME}/
    │   └── {ANNOTATION_FILE}.txt
    └── metainfo.json
```
2. Prepare all `metainfo.json` files (e.g. frames, file pattern, homography)
3. Run for each dataset:
   ```bash
   python ./src/datasets/preprocess.py --dataset {DATASET_NAME}
   ```
   Check `./datasets/{DATASET_NAME}/{SEQUENCE_NAME}/output` if there is anything missing:
   ```text
   /output/
   ├── gt_MOT/ # for motmetrics
   │   └── c{CAM}.txt
   ├── gt_train.json
   ├── gt_eval.json
   ├── gt_test.json
   └── {DETECTOR}_test.json # if you want to use other detector, e.g. yolox_test.json
   ```
5. Prepare all image frames as `{FRAME}_{CAM}.jpg` in `/output/frames`.

## Model Zoo
Download trained weights if you need, and modify `TEST.CKPT_FILE_SG` & `TEST.CKPT_FILE_TG` in `./configs/{DATASET_NAME}.yml`.
| Dataset   | Spatial Graph    | Temporal Graph   |
|-----------|------------------|------------------|
| Wildtrack | [sequence1](https://drive.google.com/file/d/1U4Qc2xHERbLUzly5gUToG2H1Rktux8mi/view?usp=sharing) | [sequence1](https://drive.google.com/file/d/17tvAeERcsy3YaB3lR2aIZYDQqKFySmVA/view?usp=sharing) |
| CAMPUS    | [Garden1](https://drive.google.com/file/d/1OCxDios5BhucUIKQSinxLIELPjfS7pXJ/view?usp=sharing)<br>[Garden2](https://drive.google.com/file/d/12lS9dOk3sWJpYo0y47Bi1P_aSsd1VnDN/view?usp=sharing)<br>[Parkinglot](https://drive.google.com/file/d/1cGzAH_DwzoR-6eifLT28dY032GjbVrds/view?usp=sharing) | [Garden1](https://drive.google.com/file/d/1a_jWuImPofqnfbO-6Wpvghtk5Nqvu6ql/view?usp=sharing)<br>[Garden2](https://drive.google.com/file/d/1Yz6C-7R0XdaaNt9dKyxTg7jFagAN7s3W/view?usp=sharing)<br>[Parkinglot](https://drive.google.com/file/d/1WiBkWAuV0KNwEFk7GQx4WVp85kcV_87w/view?usp=sharing) |
| PETS-09   | [S2L1](https://drive.google.com/file/d/1vNnNZnSupBmcv-7CnuH5p4zqNrz43YdO/view?usp=sharing) | [S2L1](https://drive.google.com/file/d/1nnumFo1ZX18WRE631nYGw6bftqOSIkDk/view?usp=sharing) |

## Training
To train our model, basically run the command:
```bash
python main.py --config_file ./configs/{DATASET_NAME}.yml
```
In `{DATASET_NAME}.yml`:
* Modify `MODEL.MODE` to 'train'
* Modify `SOLVER.TYPE` to train specific graphs.
* Make sure all settings are suitable for your device, e.g. `DEVICE_ID`, `BATCH_SIZE`.
* You can also directly append attributes after the command for convenience, e.g.:
  ```bash
  python main.py --config_file ./configs/Wildtrack.yml MODEL.DEVICE_ID "('1')" SOLVER.TYPE "SG"
  ```

## Testing
```bash
python main.py --config_file ./configs/{DATASET_NAME}.yml
```
In `{DATASET_NAME}.yml`:
* Modify `MODEL.MODE` to 'test'.
* Select what input detection you want, and modify `MODEL.DETECTION`.
    * You need to prepare `{DETECTOR}_test.json` in `./datasets/{DATASET_NAME}/{SEQUENCE_NAME}/output/` by your own first.
* Make sure all settings in `TEST` are configured.

## DEMO
### Wildtrack
<img src="https://github.com/chengche6230/ReST/blob/main/docs/ReST_demo_Wildtrack.gif" width="65%" height="65%"/>

## Acknowledgement
* Thanks for the codebase from the re-implementation of [GNN-CCA](https://github.com/shawnh2/GNN-CCA) ([arXiv](https://arxiv.org/abs/2201.06311)).

## Citation
If you find this code useful for your research, please cite our paper
<!---
```text
@InProceedings{
    author    = {},
    title     = {ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023}
}
```
--->
