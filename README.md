# Email Spam Detection

This repository contains a machine learning project aimed at detecting whether an email is **spam** or **ham** (not spam). The project uses multiple classifiers like **Logistic Regression**, **Support Vector Machines (SVM)**, **Naive Bayes**, and **Random Forest** to classify emails. After training and testing, the models are compared based on accuracy, precision, recall, and F1-score.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Fine-tuning](#fine-tuning)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to classify emails as spam or ham based on their content. The dataset is preprocessed to remove noise, and various machine learning algorithms are used to train the model. The final models are evaluated and fine-tuned to achieve high performance, and the best classifier is selected for deployment.

### Key Features:
- Preprocessing includes **cleaning** text by removing punctuation, numbers, and converting to lowercase.
- Uses **TF-IDF Vectorizer** for text representation.
- Multiple models are trained, including **SVC, Logistic Regression, Naive Bayes, and Random Forest**.
- Hyperparameter tuning is performed using **GridSearchCV** for optimal model performance.
  
## Dataset
The dataset used in this project comes from a collection of labeled emails. The labels are either:
- **Ham (0)**: Non-spam email.
- **Spam (1)**: Spam email.

You can download the dataset [here](https://www.kaggle.com/datasets/satyajeetbedi/email-hamspam-dataset).

### Dataset Structure:
- **Columns**: The dataset contains two columns:
  - `v1`: The label (ham or spam)
  - `v2`: The message content
- **Size**: ~5,572 emails.

## Modeling
The following models are used in this project:
1. **Logistic Regression**
2. **Support Vector Machine (SVC)**
3. **Naive Bayes**
4. **Random Forest**

Each model is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Punitpawar5/email-spam-detection.git
   cd email-spam-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If using Jupyter Notebooks, launch it with:
   ```bash
   jupyter notebook
   ```

## Usage
Once the project is set up, you can run the code to train the models and evaluate performance.

1. **Train models**:
   You can run the main script to train all models and print their performance metrics:
   ```bash
   python train_models.py
   ```

2. **Fine-tuning**:
   You can run the fine-tuning script to optimize hyperparameters using GridSearchCV:
   ```bash
   python fine_tune.py
   ```

3. **Test a custom email**:
   After training, you can use the trained model to classify a custom email:
   ```bash
   python classify_email.py "Your email content goes here"
   ```

## Results
After training and testing, the models achieved the following results on the test dataset:

- **Support Vector Classifier (SVC)**:
  - Accuracy: 98%
  - Precision (Spam): 0.98
  - Recall (Spam): 0.87

- **Random Forest**:
  - Accuracy: 97.9%
  - Precision (Spam): 0.99
  - Recall (Spam): 0.85

The SVC model had a slight edge in spam recall, while Random Forest excelled in spam precision.

## Fine-tuning
The **Random Forest** and **SVC** models were fine-tuned using **GridSearchCV** to find the best hyperparameters. For example, Random Forest was tuned with parameters like `n_estimators`, `max_depth`, and `min_samples_split`. 

### Example of Random Forest Hyperparameters Grid:
```python
param_grid_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
```

## Technologies Used
- **Python 3.x**
- **Pandas** for data manipulation
- **Scikit-learn** for model building and evaluation
- **TfidfVectorizer** for text representation
- **Matplotlib / Seaborn** for visualization

## Contributing
Contributions are welcome! If you'd like to improve the project or add new features, feel free to fork the repository and submit a pull request.

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b new-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Added new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin new-feature
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Screenshots:
#### Ham message example
https://github.com/Punitpawar5/email-spam-detection/blob/main/Screenshot%20(223).png

#### Spam message example
https://github.com/Punitpawar5/email-spam-detection/blob/main/Screenshot%20(224).png
---
