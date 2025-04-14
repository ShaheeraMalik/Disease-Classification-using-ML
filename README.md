```markdown
# Disease Classification Using Machine Learning

This repository contains a machine learning project aimed at classifying diseases based on various risk factors, symptoms, and signs using different encoding techniques and machine learning models. The tasks involve data preprocessing, feature extraction using TF-IDF and One-Hot Encoding, dimensionality reduction, and model evaluation using K-Nearest Neighbors (KNN) and Logistic Regression.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Task 1: TF-IDF Feature Extraction](#task-1-tf-idf-feature-extraction)
- [Task 2: Dimensionality Reduction](#task-2-dimensionality-reduction)
- [Task 3: Model Evaluation](#task-3-model-evaluation)
- [Task 4: Critical Analysis](#task-4-critical-analysis)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Project](#how-to-run-the-project)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to classify diseases into categories like **cardiovascular**, **neurological**, **respiratory**, etc., based on symptoms, risk factors, and signs. We experiment with two primary encoding techniques: **TF-IDF** and **One-Hot Encoding**, and evaluate their effectiveness in disease classification. We use K-Nearest Neighbors (KNN) and Logistic Regression (LogReg) for classification tasks and perform dimensionality reduction using PCA (for One-Hot Encoding) and SVD (for TF-IDF).

## Dataset
The dataset contains disease information, including:
- **Disease Name** (e.g., Asthma, Acute Coronary Syndrome)
- **Risk Factors** (e.g., smoking, hypertension)
- **Symptoms** (e.g., chest pain, wheezing)
- **Signs** (e.g., visible signs during physical examination)
- **Subtypes** (detailed disease subtypes)

The dataset is used for feature extraction and classification tasks.

### Files:
- `disease_features.csv`: Contains raw data including disease names, risk factors, symptoms, signs, and subtypes.
- `encoded_output2.csv`: Contains the one-hot encoded version of the dataset, where disease subtypes are represented as binary features.

## Task 1: TF-IDF Feature Extraction
This task involves extracting **TF-IDF** features from the **Risk Factors**, **Symptoms**, and **Signs** columns of the dataset. TF-IDF is used to transform these textual columns into numerical features, allowing them to be used in machine learning models. The sparsity of the TF-IDF matrix is also compared with the one-hot encoded matrix.

### Steps:
- Parse stringified lists in the dataset to actual Python lists.
- Convert each list into space-separated strings.
- Apply **TF-IDF vectorization** to these columns.
- Compare the sparsity and feature count of **TF-IDF** and **One-Hot Encoding**.

## Task 2: Dimensionality Reduction
Dimensionality reduction is applied using **Principal Component Analysis (PCA)** for the **One-Hot Encoding** matrix and **Truncated Singular Value Decomposition (SVD)** for the **TF-IDF** matrix. This step reduces the features to 2 or 3 dimensions for visualization and comparison.

### Steps:
- Apply **PCA** and **SVD** to both the **TF-IDF** and **One-Hot** encoded matrices.
- Visualize the reduced dimensions and color-code diseases by category.
- Compare the explained variance ratio of PCA and SVD.

## Task 3: Model Evaluation
In this task, we evaluate the classification performance of **K-Nearest Neighbors (KNN)** and **Logistic Regression (LogReg)** using **cross-validation**. We compare their performance based on **accuracy**, **precision**, **recall**, and **F1-score** for each encoding method (TF-IDF vs. One-Hot Encoding).

### Steps:
- Train **KNN** with different values of **k** and **distance metrics** (Euclidean, Manhattan, and Cosine).
- Train **Logistic Regression** and evaluate it using cross-validation.
- Compare performance for both models using various metrics.

### Visualizations:
- KNN F1-Score by K and distance metric.
- Comparison of TF-IDF vs One-Hot Encoding for KNN performance.
- Average F1-Score comparison between KNN and Logistic Regression.

## Task 4: Critical Analysis
This task involves a critical analysis of why **TF-IDF** might outperform **One-Hot Encoding** or vice versa, the clinical relevance of the results, and the limitations of both encoding methods. The analysis highlights the strengths and weaknesses of each method in the context of disease classification and discusses their practical applications.

### Key Points:
- **TF-IDF** vs **One-Hot Encoding**: Advantages and disadvantages.
- **Clinical Relevance**: Do TF-IDF clusters align with real-world disease categories?
- **Limitations**: Discusses the limitations of both encoding techniques.

## Installation and Setup
To run this project, you will need to have Python installed along with the required libraries. You can install the dependencies by following the steps below.

### Prerequisites:
- Python 3.x
- pip (Python package manager)

### Install Dependencies:
To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:
```text
pandas
numpy
scikit-learn
matplotlib
seaborn
IPython
```

## How to Run the Project

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/ShaheeraMalik/Disease-Classification-using-ML
   ```
2. Navigate to the project folder:
   ```bash
   cd disease-classification
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
4. Run the individual cells for each task, starting from **Task 1** to **Task 4**.

### Notes:
- The notebook is structured in such a way that each task is separate, so you can run them independently.
- Ensure that the dataset files (`disease_features.csv` and `encoded_output2.csv`) are in the same directory as the notebook.

## Contributing
If you'd like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request with your changes. Please ensure that your code adheres to the existing style guidelines and includes necessary tests and documentation.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```


