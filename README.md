# MMAE: Mean-Masked-Autoencoder-with-Flow-Mixing-for-Encrypted-Traffic-Classification


## 🛠️ Environment Setup


```bash
# Create a new conda environment
<<<<<<< HEAD
<<<<<<< HEAD
conda create -n MMAE python=3.8.18 -y
=======
conda create -n MMAE python=3.8.18
>>>>>>> 265cb7a5e28f98eb1238b034ce7dfcf8efcb1bdc
=======
conda create -n MMAE python=3.8.18
>>>>>>> 265cb7a5e28f98eb1238b034ce7dfcf8efcb1bdc

# Activate the environment
conda activate MMAE

# Install the required packages
pip install -r requirements.txt
```



## 📦 Data Preparation

The datasets used for this project, including both the processed **pre-training** and **fine-tuning** datasets, are securely stored and shared via Quark Drive.

- **Download Link**: [Quark Drive Link](https://pan.quark.cn/s/662574b2c536)

Please download and extract the dataset to your local machine or server before running the code.

## 🏃‍♂️ Running the Code

### 1. Pre-training

To pre-train the MMAE model, use the following command. Make sure to update the paths to match your local dataset and output directories.

```bash
CUDA_VISIBLE_DEVICES=0 python src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes <num-class> \
    --finetune <your-pretrained-checkpoint-path> \
    --data_path <your-finetune-data-dir> \
    --output_dir <your-finetune-output-dir> \
    --log_dir <your-finetune-log-dir> \
    --model mmae_classifier \
    --no_amp
```

### 2. Fine-tuning (including evaluation)

For fine-tuning and evaluating on specific downstream tasks, run the command below. You need to specify the pretrained checkpoint path and the correct <num-class> for the target dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python src/fine-tune.py \
    --blr 2e-3 \
    --epochs 120 \
    --nb_classes <num-class> \
    --finetune <your-pretrained-checkpoint-path> \
    --data_path <your-finetune-data-dir> \
    --output_dir <your-finetune-output-dir> \
    --log_dir <your-finetune-log-dir> \
    --model mmae_classifier \
    --no_amp
```
