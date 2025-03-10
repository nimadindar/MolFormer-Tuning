# MolFormer-Tuning 

## 🚀 Project Title: **Fine-Tuning and Data Selection for Lipophilicity Prediction Using MoLFormer**  

### 📌 **About**  
This project explores the impact of different **data selection strategies** and **fine-tuning techniques** on lipophilicity prediction using **MoLFormer**, a transformer-based chemical language model. The goal is to enhance model performance by selecting the most informative molecular samples and optimizing the learning process.  

### 📂 **Project Structure**  
```
📦 MolFormer-Tuning 
 ┣ 📂 datasets/                # External dataset and selected data using different data selection methods
 ┣ 📂 notebooks/               # Jupyter notebook for Task1
 ┣ 📂 reports/                 # Final project report
 ┣ 📂 src/                     # Models and utils used in tasks
 ┣ 📂 Task2/                   # Scripts and results for Task2
 ┣ 📂 Task3/                   # Scripts and results for Task3
 ┣ 📝 requirements.txt         # List of dependencies  
 ┣ 📝 README.md                # Project documentation (this file)  

```

---

## 🛠 **Setup & Installation**  
### **1️⃣ Install Dependencies**  
Before running the project, install the required Python libraries: (Python Version =3.12)  
```bash
pip install -r requirements.txt
```


### **2️⃣ Run Data Selection Experiments**  
To perform data selection using different strategies (**Influence Functions, Uncertainty-Based, S2L**), use:  
```bash
python -m Task2.Influence-lissa 
python -m Task3.Uncertainty-DS
python -m Task3.S2L-DS
```
The resulted selected data will be saved to 📂 datasets/ .

### **3️⃣ Train & Fine-Tune the Model**  
To train MoLFormer with the selected data using different fine-tuning techniques:  
```bash
python -m Task2.train-model
python -m Task3.BitFit
python -m Task3.LoRA
python -m Task3.IA3
python -m Task3.train_multiInput
```
Task2.train-model which is MLM with Regression Head is trained using all of the data selection methods. Uncertainty-based data selection has the best performance in this model. So BitFit, LoRa, and IA3 fine-tuning methods were traiimplemented using data selected by Uncertainty-based method. The resulted loss curve for each method will be used in the same directory which the script is.

---

## 🧪 **Experiments & Findings**  
- **Data Selection Strategies Evaluated**:  
  - **Uncertainty-Based Selection (Best Performance ✅)**  
  - **Small-to-Large (S2L) Selection**  
  - **Influence Functions**  
- **Fine-Tuning Methods Compared**:  
  - **Masked Language Modeling (MLM)**  
  - **BitFit (Best Performance ✅)**  
  - **LoRA**  
  - **(IA)³**  
- **Key Findings**:  
  - **Uncertainty-Based Selection outperformed other strategies**, improving generalization by prioritizing underrepresented molecular samples.  
  - **BitFit fine-tuning** led to significant performance improvements over baseline methods.  
  - A **multi-input model** integrating **domain knowledge (RDKit descriptors)** was explored for further performance enhancement.  

---

## 🤝 **Contributors**  
- **Nima DindarSafa** (*Saarland University*)  
- **Samira Abeidni**  (*Saarland University*)

---

## 📧 **Contact**  
For any questions or issues, feel free to reach out at:  
📩 Email: {nidi00002, saab00012}@stud.uni-saarland.de