# PhyAssist source code
## Structure
1. **adapter_mentalLLama_DR**:  
    Fine-tuning Adapter on the mentalLLama dataset DR
2. **adapter_mentalLLama_dreaddit**:  
 Fine-tuning Adapter on the mentalLLama dataset dreaddit.
3. **adapter_mentalLLama_Final**:  
 Adapter for on mentalLLama dataset Irf, MultiWD,
SAD
4. **adapter_mentalLLama_nbertagnolli**:  
 Fine-tuning Adapter on the mentalLLama dataset nbertagnolli
5. **MentalLLaMA_dataset**:  
 The testbench for the mentalLLama dataset.
6. **nbertagnolli_dataset**:  
 The nbertagnolli dataset for training.
7. demo.ipynb:   
The web demo launching our model.
8. **gen_csv.py**:  
 clean the training data and make the data matching the format of the base model.
9. **Mistral_Fine_Tune.ipynb**:  
 The fine-tuning code for the base model.
10. **Mistralai_Test.ipynb**:  
 The testing code for the base model.

## Dataset and Testbench
You may need download the IMHI dataset 
```
git clone https://github.com/SteveKGYang/MentalLLaMA.git
```
For the nbertagnolli dataset, you can download it using 
```
from datasets import load_dataset
load_dataset(data_path, split="train")
```
When you trying to use the testbenchs in the mentalLLama dataset, you need to follow the Readme.md in MentalLLaMA_dataset to download the classifier model to test the accuracy and download the Bart model to calculate the Bart score.

## Data Preprocessing
The data preprocessing is in the gen_csv.py. You can use it to clean the data and make the data matching the format of the base model. The output is a csv file with a column named "text".

You can use it by
```
python gen_csv.py --origin_path <source_data_path> --new_path <output_path>
```
Before using, you need to change some code in the gen_csv.py to adapt to your data format. 
* The first place(line 22):
```python
data_content = "<s>[INST]"+ [The Question Columns] + " [/INST]  Anwser:" + [The Answer Columns] + "</s>"
```
* The second place(line 32-34):
```python
if(len(content) != [Your Data Columns Number]):
    continue
dirty_data = Columns[0] + ... + Columns[-1]
```

## Data Fine-tuning
The fine-tuning code is in the **Mistral_Fine_Tune.ipynb**. You can use it to fine-tune the base model. The output is a adapter checkpoint. You can change the output path in the code.

## Data Testing
The testing code is in the **mistralai_test.ipynb**. You can use it to test the accuracy of the base model. You can change the output path in the code.

In the testing code, there is some command to use the testbench in the mentalLLama dataset. 
If the command is facing some error, you can change the command according to the Readme.md in the **MentalLLaMA_dataset**.

## The structure of the MentalLLaMA_dataset
1. model_output:
The output of our fine-tuning model on the test_data.
2. test_data:
the test data of the mentalLLama dataset. The model output is generated by our model using the data in **test_complete**.
3. src: 
The source code of the testbench.
* **IMHI.py** using a model provided by the user to generate the output with the test data.
* **label_inference.py** using the model output to calculate the **accuracy** of model's prediction. When using it, this code will ask you to download the classifier model the mentalLLama group has trained for each dataset they provided. 
* **socre.py** Calculate score on some benchmarks of the test data, which is used to judge the **quality** of the model prediction. When using it, this code will ask you to download some models of these benchmarks.

## demo.ipynb
The demo.ipynb is the web demo launching our model using gradio. You can use it to test our model. 










