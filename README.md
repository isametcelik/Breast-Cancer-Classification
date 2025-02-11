**Breast Cancer Classification using Machine Learning**

**Overview**

This project implements a machine learning model to classify breast cancer tumors as benign or malignant. The dataset used is the Breast Cancer Wisconsin Dataset. The primary goal is to preprocess the data, apply dimensionality reduction techniques, and use K-Nearest Neighbors (KNN) to classify tumors efficiently.

**Features**

**Exploratory Data Analysis (EDA):**

Data visualization and correlation analysis.

Outlier detection using Local Outlier Factor (LOF).

**Preprocessing:**

Data standardization with StandardScaler.

Dimensionality reduction using Principal Component Analysis (PCA) and Neighborhood Components Analysis (NCA).

**Model Training:**

Implementation of K-Nearest Neighbors (KNN).

Hyperparameter tuning using GridSearchCV.

**Performance Evaluation:**

Confusion matrix and accuracy score analysis.

Visualization of decision boundaries for PCA and NCA.

**Installation**

To run this project, install the necessary dependencies:

'pip install numpy pandas matplotlib seaborn scikit-learn'

**Usage**

**Clone the repository:**

'git clone https://github.com/isametcelik/breast-cancer-classification.git
cd breast-cancer-classification'

**Run the Python script:**

'python main.py'

**Results**

Best hyperparameters for KNN are selected using GridSearchCV.

PCA and NCA are used for feature reduction and visualization.

Classification performance is analyzed using accuracy score and confusion matrix.

**Dataset**
'The dataset used is Breast Cancer Wisconsin (Diagnostic) Data Set, available [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).'

**Author**

'Ismail Samet Celik - Computer Programming Student at Toros University'
