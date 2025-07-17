# 🧠 Resume Classifier — NLP + Machine Learning Project

Automatically classifies resumes into job categories using NLP techniques and multiple machine learning models.

---

## 📁 Project Structure


 Resume-Classifier-ML-Project/
├── data/
│ ├── download_data.ipynb
│ └── resume-dataset/
│ └── UpdatedResumeDataSet.csv
├── notebook/
│ ├── 01_EDA_Resume.ipynb
│ ├── 02_Preprocessing_Resume.ipynb
│ ├── 03_Model_Training_Evaluation.ipynb
│ └── 04_Tuning_Model.ipynb
├── models/
│ ├── X_train.joblib
│ ├── X_test.joblib
│ ├── y_train.joblib
│ └── y_test.joblib
├── images/
│ ├── 0.png ... 4.png
├── .gitignore
└── README.md


---

## 📊 Dataset

- **Source**: The dataset is downloaded from Kaggle.
- **File**: `UpdatedResumeDataSet.csv`
- **Target Variable**: Category of job role.
- **Text Column**: Resume content.

---

## 🔍 Project Workflow

### 1. EDA (Exploratory Data Analysis)

- Checked class distribution and common keywords.
- Visualized most frequent words using word clouds and bar charts.

### 2. Preprocessing

- Removed URLs, special characters, stopwords, etc.
- Encoded categorical labels.
- Applied TF-IDF vectorization.
- Split dataset into train and test sets.
- Saved processed data using `joblib`.

### 3. Model Training & Evaluation

Evaluated several classifiers using accuracy, confusion matrix, and classification report:

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.53     |
| Random Forest       | 0.76     |
| Naive Bayes         | 0.35     |
| K-Nearest Neighbors | 0.82     |
| One-vs-Rest (SVC)   | 0.79     |
| One-vs-Rest (KNN)   | 0.82     |



## 🔧 Advanced Tuning & Local SVM

In `04_Tuning_Model.ipynb`, we applied two main optimization techniques:

### A. Hyperparameter Tuning with Optuna

- Optimized `C`, `kernel`, and `gamma` parameters for SVC.
- Improved generalization:
  - **Training Accuracy**: dropped slightly (92% → 88%) to reduce overfitting.
  - **Testing Accuracy**: increased (82% → 85%).


> 🔍 Note: The Optuna optimization code was removed after applying the best


We implemented a **Localized SVM** approach:

- For every resume vector (training or test):
  - Find the `k` nearest neighbors using `sklearn.NearestNeighbors`.
  - Train a dedicated `SVC` model only on those neighbors.
  - Use it to predict the class of the target point.
- If training fails (e.g., only one class), we fallback to the global majority class.

> This method captures **local patterns** in the data and is especially useful in **non-linear or imbalanced** datasets. However, it is computationally expensive, as it trains an SVM for every single point.


---

## 📈 Visualizations

- All plots and evaluation graphs (e.g., word clouds, confusion matrices) are saved in the `images/` folder.


## 📌 Future Improvements

- Use advanced NLP models like **BERT**, **spaCy**, or **LlamaIndex**.
- Segment resume content into sections (e.g., skills, education) before classification.
- Build a **Streamlit** or **Flask** app to deploy the model.


## 📬 Contact

For any feedback or suggestions, feel free to reach out:

**Name**: Younes Elshafi  
**Email**: [younes.ai.dev@gmail.com](mailto:younes.ai.dev@gmail.com)




