# MolFormer-Tuning


A project focused on tuning **MoLFormer** model using external datasets, data selection methids, and advanced fine-tuning techniques.

## Overview
This repository explores different strategies for improving **MoLFormer-XL**'s performance on the **MoleculeNet Lipophilicity dataset**. The project includes:
- **Influence Functions with LiSSA Approximation** to select high-impact external data points.
- **Alternative Data Selection Methods** to optimize training.
- **Fine-tuning Techniques** such as BitFit, LoRA, and iA3.

## Repository Structure
```
MolFormer-Tuning/
│── data/                    # Dataset and preprocessing scripts
│── models/                  # Pretrained and fine-tuned models
│── notebooks/               # Jupyter notebooks for experiments
│── src/                     # Core source code for training & evaluation
│── results/                 # Logs, metrics, and analysis outputs
│── requirements.txt         # Required dependencies
│── README.md                # Project documentation
```

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/nimadindar/MolFormer-Tuning.git
cd MolFormer-Tuning
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Running the Model Training
```bash
python src/train.py --config configs/config.yaml
```

### Running Influence Function Analysis
```bash
python src/influence_analysis.py --data data/External-Dataset_for_Task2.csv
```

### Fine-tuning the Model
```bash
python src/fine_tune.py --method LoRA
```

## Results
Key findings and results are stored in the `results/` directory. Notable metrics and insights will be documented here.

## Contribution
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

## Contact
For questions or collaborations, please reach out via GitHub issues or email: **your-email@example.com**

## License
This project is licensed under the MIT License. See `LICENSE` for details.
