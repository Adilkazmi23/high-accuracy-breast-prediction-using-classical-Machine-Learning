
---

# High-Accuracy Breast Cancer Prediction using Classical Machine Learning

Project Overview

This project focuses on the early and accurate detection of breast cancer (Benign vs. Malignant) using patient diagnostic data. By leveraging **classical machine learning algorithms** and rigorous data preprocessing, this model achieves high predictive accuracy, making it a reliable tool for medical data analysis.

## 📊 Dataset

The project utilizes the **Wisconsin Breast Cancer (Diagnostic) Dataset**. It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, such as:

* Radius, Texture, Perimeter, and Area
* Smoothness, Compactness, and Concavity
* Symmetry and Fractal Dimension

 Key Features

* **Data Cleaning & Scaling:** Implementation of `StandardScaler` to normalize feature ranges for better convergence.
* **Exploratory Data Analysis (EDA):** Visualizing correlations and feature distributions using Seaborn and Matplotlib.
* **Model Comparison:** Comparison of multiple classical algorithms:
* Logistic Regression
* Support Vector Machines (SVM)
* Random Forest Classifier
* K-Nearest Neighbors (KNN)


* **High Performance:** Optimized hyperparameters to reach peak accuracy (up to 98-99%).

## 🛠️ Installation & Usage

1. **Clone the repository:**
```bash
git github.com/Adilkazmi23/high-accuracy-breast-prediction-using-classical-Machine-Learning.git

```


2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```


3. **Run the Notebook:**
Open `breast_cancer_prediction.ipynb` in Jupyter or Google Colab to see the analysis and results.
 Results

The models were evaluated based on:

* **Accuracy:** Overall correctness of the model.
* **Precision & Recall:** Critical for minimizing false negatives in medical diagnosis.
* **F1-Score:** The harmonic mean of precision and recall.

| Model | Accuracy |
| --- | --- |
| Logistic Regression | ~98% |
| SVM | ~97% |
| Random Forest | ~96% |

 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/Adilkazmi23/high-accuracy-breast-prediction-using-classical-Machine-Learning/issues).

---

**Author:** [Adil Sayyed](https://www.google.com/search?q=https://github.com/Adilkazmi23)
