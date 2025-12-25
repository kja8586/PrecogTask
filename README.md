# Can you break the CAPTCHA

## Project overview
Here there are three tasks:
Task0 : Synthetic data generation
Task1 : Closed-set classification
Task2 : Open-set image to text generation

This repository contain all the code files that have been used to complete the tasks

## Repo structure
PrecogTask/
│── Task0/           Notebook used for data generation
│── Task1/           Script,code and log files for classification task
│── Task2/           Script,code and log files for generation task
│── README.md        Project documentation
│── crnn_final.pth   Task2 saved model

## How to run the project

### 1. Clone the repository
git clone https://github.com/kja8586/PrecogTask.git
### 2. Navigate to the repository
cd PrecogTask
### For Task0
Step1 : Upload the notebook present in the Task0 folder to google drive.
Step2 : Run the first cell of the notebook and then restart session as we need some downgraded modules.
Step3 : For generation of Hard Set and Bonus Set make sure you also upload fonts and background folder into you drive make give the correct path
Step4 : Now run all cell you will get all three dataset.

### For Task1 and Task2
If you have any HPC or local sever then replace task1 with the compatible script and then submit the script else give just run the files using
python Task1/task1.py
python Task2/task2.py
