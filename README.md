
# 🔊 UrbanSound8K Audio Classification

UrbanSound8K is a sound classification project that identifies various urban audio events such as sirens, drilling, dog barking, and more using machine learning models. This project compares several classifiers to determine which performs best on environmental sound recognition.

---

## 🎯 Objective

To build and compare multiple machine learning models that classify 10 categories of urban sounds using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

---

## 📂 Dataset Overview

The UrbanSound8K dataset consists of **8,732 labeled sound excerpts** from 10 classes:

* Air conditioner
* Car horn
* Children playing
* Dog bark
* Drilling
* Engine idling
* Gun shot
* Jackhammer
* Siren
* Street music

---

## 🛠️ Tools & Libraries

* Python, Pandas, NumPy
* Scikit-learn (Logistic Regression, SVM, KNN, etc.)
* XGBoost, LightGBM
* Matplotlib, Seaborn
* ROC, AUC, Confusion Matrices

---

## 🧠 Models Tested

| Model                  | Accuracy |
| ---------------------- | -------- |
| Logistic Regression    | 0.25     |
| Decision Tree          | 0.92     |
| Random Forest          | 0.80     |
| Gradient Boosting      | 0.79     |
| Support Vector Machine | 0.32     |
| K-Nearest Neighbors    | 0.68     |
| Naive Bayes            | 0.17     |
| XGBoost                | 0.89     |
| LightGBM               | 0.89     |
| Neural Network (MLP)   | 0.47     |

> ✅ **Best performers**: Decision Tree, LightGBM, and XGBoost

---

## 📊 Visualizations

* 📉 **Confusion Matrix**: Model-wise confusion plots for class accuracy.
* 📈 **ROC Curve**: Multi-class ROC using one-vs-rest strategy.
* 📊 **Bar Charts**: Predicted class distribution.

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* ROC-AUC (macro and weighted)

---

## 🚀 How to Run

1. **Install dependencies**:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
   ```

2. **Run your training script**:
   Make sure `UrbanSound8K.csv` is accessible in your path.

---

## 💡 Future Improvements

* Use MFCC features from raw audio for improved performance.
* Explore CNN/LSTM models using spectrograms.
* Hyperparameter tuning for ensemble models.

---