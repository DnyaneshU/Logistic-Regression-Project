# Vehicle Breakdown Prediction using Logistic Regression

This project focuses on predicting whether a vehicle is likely to **require maintenance** or **not** using a Logistic Regression model. It's designed to help fleet operators, service centers, or individual users proactively manage vehicle health based on historical and operational data.

---

## Contents

- [Project Overview](#project-overview)
- [Dataset Features](#dataset-features)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [Future Enhancements](#future-enhancements)
- [Requirements](#requirements)

---

## Project Overview

The objective is to classify vehicles into:

- **Class 0** → No Maintenance Needed  
- **Class 1** → Maintenance Required

We use various features such as mileage, vehicle age, fuel type, brake condition, battery status, etc., to make this prediction. The model is trained using **Logistic Regression**, a widely used classification algorithm.

---

## Dataset Features

The dataset contains 20+ features related to vehicle condition and service history. You can find the complete dataset in the files section itself or the link to the dataset is here: https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data

---

## Data Preprocessing

- **Missing Value Handling** using `SimpleImputer`
- **One-Hot Encoding** for categorical variables
- **Date columns dropped** for simplicity
- **Train/Test split** of 80/20
- Data fed into a **pipeline** combining preprocessing and the logistic regression model

---

## Exploratory Data Analysis (EDA)

- **Correlation Heatmap** to check relationships between features and the target  
- **Distribution Plot** to examine the target class balance  
- **Pairplots / Scatterplots** to visually assess patterns in key variables

---

## Model Training

- Logistic Regression via `sklearn.linear_model`
- Train/Test split: 80/20
- Feature scaling used where required
- Model integrated into a **pipeline** with preprocessing steps

---
## Visualizations

### 🔹 Confusion Matrix Heatmap

![Confusion Matrix](https://github.com/DnyaneshU/Logistic-Regression-Project/blob/main/Plots/confusion_matrix.png)

Here, the points along these axes represent: 
- **True Positives**: Vehicles that actually needed maintenance and were correctly predicted — the model is highly effective at catching issues.
- **True Negatives**: Vehicles correctly predicted as not needing maintenance.
- **False Positives**: A few vehicles were flagged as needing maintenance when they didn't — better safe than sorry in predictive maintenance.
- **False Negatives**: Slightly more critical — vehicles that needed maintenance but were missed.

Also, some Key Takeaways from the graph above suggests:

- The model has a **high accuracy**, with very few false negatives.
- **Precision and recall for Class 1 (Needs Maintenance)** are excellent, making it reliable for safety-focused applications.
- Suitable for real-world use where **proactive detection** is crucial.
  
---

## Conclusion

- Logistic Regression proved effective for binary classification in vehicle diagnostics.  
- Model performance is solid and generalizes well on unseen data.  
- Useful for **preventive maintenance systems** in transportation, logistics, and automotive industries.  
- Can be enhanced further using more temporal data or advanced models like Random Forest or Gradient Boosting.

---

## Future Enhancements

Some additional details that could be worked on this project are:
- Include time-series analysis using service dates  
- Use ensemble models to boost classification accuracy  
- Build a real-time dashboard using Streamlit  

---

## Requirements

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`

Install via:

```bash
pip install -r requirements.txt
