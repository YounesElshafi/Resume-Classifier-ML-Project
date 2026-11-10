# Résumé Category Classifier

A notebook project that compares NLP models for classifying résumé text into job categories. It covers exploratory analysis, TF-IDF features, baseline classifiers, and an experimental nearest-neighbor SVM. Its scope is classification of the bundled dataset.

## Reproduce the experiment

Use Python 3.11 and install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook notebooks
```

Run the notebooks from the `notebooks/` directory, in this order. NLTK downloads its English stopword list on the first run.

| Notebook | What it does |
| --- | --- |
| [01 · Exploratory analysis](notebooks/01_EDA_Resume.ipynb.ipynb) | Explores the text and category distributions. |
| [02 · Preprocessing](notebooks/02_Preprocessing_Resume.ipynb) | Deduplicates the text, makes a stratified split, and learns the TF-IDF vocabulary and weights on the training split. It writes the feature matrices and labels to `models/`. |
| [03 · Baseline evaluation](notebooks/03_Model_Training_Evaluation.ipynb) | Compares logistic regression, random forest, multinomial Naive Bayes, KNN, and one-vs-rest models on the held-out split. |
| [04 · Local SVM](notebooks/04_hyperparamter_Model.ipynb) | Fits an SVM on a small neighborhood from the training data for each test example. It uses the training majority class for single-class neighborhoods. |

Start with notebook 02 to regenerate the feature files before running either evaluation. The metrics printed by notebooks 03 and 04 come from the updated split and preprocessing pipeline. The [images](images/) folder contains plots from earlier runs.

On the 34-example held-out split, one-vs-rest KNN classified 29 examples correctly (85.3%) and the local SVM classified 28 correctly (82.4%). These results describe this small dataset split.

## Dataset

[UpdatedResumeDataSet.csv](data/resume-dataset/UpdatedResumeDataSet.csv) contains the `Resume` text and `Category` label. The separate [download notebook](data/download_data.ipynb) records how the dataset was obtained.
