# Spatial gene expression at single-cell resolution from histology using deep learning with GHIST

For more details, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1).

The increased use of spatially resolved transcriptomics provides new biological insights into disease mechanisms. However, the high cost and complexity of these methods are barriers to broad clinical adoption. Consequently, methods have been created to predict spot-based gene expression from routinely-collected histology images. Recent benchmarking showed that current methodologies have limited accuracy and spatial resolution, constraining translational capacity. Here, we introduce GHIST, a deep learning-based framework that predicts spatial gene expression at single-cell resolution by leveraging subcellular spatial transcriptomics and synergistic relationships between multiple layers of biological information. We validated GHIST using public datasets and The Cancer Genome Atlas data, demonstrating its flexibility across different spatial resolutions and superior performance. Our results underscore the utility of in silico generation of single-cell spatial gene expression measurements and the capacity to enrich existing datasets with a spatially resolved omics modality, paving the way for scalable multi-omics analysis and new biomarker discoveries.  

![alt text](Figure1.png)


## Installation

> **Note**: A GPU with 24GB VRAM is strongly recommended for the deep learning component, and 32GB RAM for data processing.
We ran GHIST on a Linux system with a 24GB NVIDIA GeForce RTX 4090 GPU, Intel(R) Core(TM) i9-13900F CPU @ 5.60GHz with 32 threads, and 32GB RAM.

1. Clone repository:
```sh
git clone https://github.com/SydneyBioX/GHIST.git
```

2. Create virtual environment:
```sh
conda create --name ghist python=3.10
```

3. Activate virtual environment:
```sh
conda activate ghist
```

4. Install dependencies:
```sh
cd GHIST
pip install -r requirements.txt
```

Typically installation is expected to be completed within a few minutes.


## Demo

The demo dataset is based on publicly available data provided by 10x Genomics (In Situ Sample 2): https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast. We will use a subset of the data and a previously saved checkpoint as a short demo.


### Running the demo

1. Unzip the file `data_demo/images.zip` and place the 3 `.tif` files under `data_demo`:
```sh
unzip data_demo/images.zip -d data_demo/
```

2. Download saved model checkpoint:
```sh
gdown --folder https://drive.google.com/drive/folders/1ecTOXmSeQU9v8aKYniQQab2QqkOlIl8u?usp=sharing
```

3. Train from checkpoint:
```sh
python train.py --use_sc --demo
```

4. Validation mode:
```sh
python test.py --use_sc --demo --mode val
```

5. Testing mode:
```sh
python test.py --use_sc --demo --mode test
```

6. Inference mode (does not require labels):
```sh
python inference.py --use_sc --demo --no_normstain
```

The predictions are stored in ``experiments/{timestamp}/{mode}_output``, and the csv files contains the predicted gene expressions for each cell, where the index is the cell ID that corresponds to the IDs from the nuclei segmentation image, and the columns are the genes. Demo is expected to be completed within a few minutes.


## Running GHIST:

Create a config file under ``./configs``, that contain the paths to data under ``data_sources``, and other parameters. Please see the demo config file as an example.


### Training the model
```sh
python train.py --config_file configs/FILENAME.json --resume_epoch None --fold_id FOLD
```
- ``--resume_epoch`` specifies whether to train from scratch or resume from a checkpoint, e.g., ``--resume_epoch 10`` to resume from the saved checkpoint from epoch 10
- ``--fold_id`` specifies the cross-validation fold
- ``--use_sc`` whether to use single cell reference as input (optional)


### Predicting from the trained model

```sh
python test.py --config_file configs/FILENAME.json --epoch EPOCH --mode MODE --fold_id FOLD --alpha 2
```
- ``--epoch`` specifies which epoch to test, e.g., ``--epoch 40`` to use the model from epoch 40, ``--epoch -1`` to use the model from the last saved epoch
- ``--mode`` can be ``val`` or ``test``
- ``--alpha`` hyperparameter for neighbourhood-based refinement, where larger values lead to more refinement
- ``--use_sc`` whether to use single cell reference as input (optional)
- ``--is_invasive`` whether it is invasive breast cancer (optional)

For inference only without labels available and its additional flags:
```sh
python inference.py --config_file configs/FILENAME_inference.json --test_epoch EPOCH --fold_id FOLD --alpha 2
```
- ``--no_normstain`` if not using stain normalisation. Otherwise, specify ``data_sources.fp_stain_target`` in the config file, that is a path to a normalisation target image 


### Output cell expressions

Predicted gene expressions individual cells may be found in the experiment directory, e.g.: ``experiments/fold1_2024_06_21_20_44_56/test_output/epoch_50_expr.csv``
An example output csv file is provided, please see ``example_output.csv``

## Citation

If GHIST has assisted you with your work, please kindly cite our paper:

- https://www.biorxiv.org/content/10.1101/2024.07.02.601790v1
