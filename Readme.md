<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="images/pipeline.png" width="600" /></a><br>
  <b>PhyAssist: Fine-Tuning Large Language Models into Personalized Psychological Counseling Assistants</b><br>
</h1>


PhyAssist is a fine-tuned large language model designed to serve as an intelligent assistant for psychotherapists. We use the ``Mixtral-8x7B-Instruct-v0.1`` as our backbone model and fine-tune it on the **IMHI dataset** to adapt it to the psychological counseling domain. It can help mental health professionals handle patient consultations more efficiently through natural conversations, providing comprehensive symptom analysis and professional diagnostic suggestions. The model is trained on psychological counseling datasets to adapt large language models for specialized clinical support.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)

## Introduction

For a detailed overview of this project, including the design ideas and pending tasks, please refer to the [Introduction.md](Introduction.md) file.

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch. We test the code on Python 3.8.10 and PyTorch 2.1.0. If you only have CPU or older versions of these libraries, you may need to modify the code accordingly. 

### Installation

1. Clone the repository:

```bash
git clone https://github.com/PhyAssist-project/PhyAssist_source.git
cd PhyAssist_source
```

2. Create an environment and install dependencies:

```bash
conda create -n phyassist python=3.8.10 -y
conda activate phyassist
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

You may encounter errors when using the `Trainer` from 🤗 Transformers, for example:

```python
from transformers import Trainer
```
This issue may be related to the apex package. You can resolve it by uninstalling apex:
```python
pip uninstall -y apex
```

## Usage

We provide a simple demo notebook to quickly experience **PhyAssist**.

### Run the demo notebook

Open the notebook in Jupyter or VSCode:

```bash
jupyter notebook demo.ipynb
```

This will guide you through loading the fine-tuned model and starting the Gradio interface.

### Launch the Gradio demo

After running the notebook, a Gradio app will start automatically.
You can interact with **PhyAssist** in your browser at:

```
http://localhost:7860
```

Use this interface to simulate psychological consultation scenarios and explore how **PhyAssist** assists psychotherapists with patient requests.

## Training

### Dataset Download and Preprocessing

#### 1. IMHI dataset
We use the **IMHI dataset** to fintune the base model. Please follow the instructions in the [MentalLLaMA_dataset README](https://github.com/SteveKGYang/MentalLLaMA) to download the IMHI dataset.

When using the testbenches provided in the **MentalLLaMA** dataset to test the accuracy of the model, you need to download the classifier model(for accuracy evaluation) and the BART model(for BART score calculation). Instrcutions are also provided in the [MentalLLaMA_dataset README](https://github.com/SteveKGYang/MentalLLaMA).

After downloading the dataset, you can preprocess the data by running the `gen_csv.py` script. This will generate a CSV file with a single column named `text`.

```bash
python gen_csv.py --origin_path <source_data_path> --new_path <output_path>
```

You may need to slightly modify `gen_csv.py` to adapt it to your dataset format:

1. **Line 22**: Define how to concatenate question and answer columns.

```python
data_content = "<s>[INST]" + [Question Columns] + " [/INST] Answer: " + [Answer Columns] + "</s>"
```

2. **Lines 32–34**: Adapt to the number of columns in your dataset.

```python
if len(content) != [Your Data Column Number]:
    continue
dirty_data = Columns[0] + ... + Columns[-1]
```

This ensures that the generated CSV is properly aligned with our training pipeline.


#### 2. nbertagnolli dataset

For the **nbertagnolli dataset**, you can directly load it via 🤗 Datasets:

```python
from datasets import load_dataset
data_path = "nbertagnolli_dataset/counsel_chat.csv"
load_dataset(data_path, split="train")
```

### Training Script

To fine-tune the base model, you can use the `Mistral_Fine_Tune.ipynb` notebook. This notebook will guide you through the fine-tuning process.

## Results

We evaluate the fine-tuned **PhyAssist** model using two complementary approaches:

1. **General LLM Capability Evaluation**  
   We compare PhyAssist with other large language models on common benchmarks to assess its overall language modeling capabilities.  
   ![General Capability Results](images/result1)

2. **IMHI Test Bench for Emotional Counseling**  
   We further validate PhyAssist on the **IMHI test bench**, which specifically measures performance in emotional counseling scenarios.  
   ![Emotional Counseling Results](images/result2)
