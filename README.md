# BGPredictor

A machine learning model for predicting blood glucose levels one hour into the future using time-series data from participants with type 1 diabetes.

## Table of Contents

- [Introduction](#introduction)
- [The Challenge](#the-challenge)
- [Data Description](#data-description)
- [Approach](#approach)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Modeling](#modeling)
    - [Stacked Ensemble Model](#stacked-ensemble-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction

This project aims to predict future blood glucose levels using historical physiological and behavioral data. Accurate forecasting of glucose levels can significantly aid individuals with type 1 diabetes in managing their condition and preventing adverse events such as hypoglycemia or hyperglycemia.

## The Challenge

The **BrisT1D Blood Glucose Prediction Competition** on Kaggle challenges participants to develop models that predict blood glucose levels one hour ahead, based on a rich dataset collected from individuals with type 1 diabetes.

Link to the competition: [BrisT1D Blood Glucose Prediction Competition](https://kaggle.com/competitions/brist1d)

## Data Description

The dataset includes:

- **Time-Series Measurements**: Blood glucose levels (`bg`), insulin doses (`insulin`), carbohydrate intake (`carbs`), heart rate (`hr`), steps (`steps`), and calories burned (`cals`), recorded every 5 minutes.
- **Activity Types**: Categorical variables indicating different types of physical activities.
- **Demographic Information**: Participant identifiers.

**Files:**

- `train.csv`: Training data with the target variable `bg+1:00` (blood glucose level one hour ahead).
- `test.csv`: Test data without the target variable.
- `sample_submission.csv`: Sample format for submission.

**Note**: The dataset files are not included in this repository due to licensing restrictions.

## Approach

To tackle the prediction task, the following steps were undertaken:

### Data Preprocessing

1. **Handling Missing Values**:
   - **Numerical Variables**: Imputed missing values using the median.
   - **Categorical Variables**: Replaced missing values with a new category `'missing'`.

2. **Encoding Categorical Variables**:
   - Used `LabelEncoder` to convert categorical variables into numerical format, ensuring consistent encoding across training and test sets.

3. **Feature Engineering**:
   - Extracted time features (`hour`, `minute`) from the `time` column.
   - Created lag features for variables like `bg`, `insulin`, `carbs`, `hr`, `steps`, and `cals` for the past hour.
   - Generated moving averages and interaction features to capture underlying patterns.

4. **Feature Scaling**:
   - Applied `RobustScaler` to numerical features to reduce the impact of outliers.

### Exploratory Data Analysis

- Analyzed the distribution of blood glucose levels.
- Examined correlations between features and the target variable.
- Visualized time-series patterns for individual participants to understand trends and anomalies.

### Modeling

#### Stacked Ensemble Model

To achieve the best predictive performance, a stacked ensemble model was developed using Scikit-Learn's `StackingRegressor`:

- **Base Models**:
  - **XGBoost Regressor**: Captures complex nonlinear relationships and is efficient for structured data.
  - **CatBoost Regressor**: Handles categorical features natively and is robust to overfitting.

- **Meta-Model**:
  - **RidgeCV Regressor**: Used as the final estimator to learn the optimal combination of base model predictions.

- **Implementation**:
  - Individually hyperparameter-tuned both base models (XGBoost and CatBoost) for optimal performance.
  - Set up the stacking ensemble with the tuned base models and `RidgeCV` as the meta-model.
  - Trained the stacking ensemble on the training data.
  - The ensemble model learns how to best combine the predictions of the base models to improve overall accuracy.

- **Advantages**:
  - **Diversity in Modeling Techniques**: Combining gradient boosting frameworks (XGBoost and CatBoost) with a linear model (RidgeCV) helps capture a wide range of data patterns.
  - **Improved Generalization**: The stacking approach reduces overfitting by leveraging the strengths of different algorithms.

## Results

- **Evaluation Metric**: Root Mean Squared Error (RMSE) on the validation set.
- **Performance**:
  - The stacked ensemble model achieved an RMSE of **2.7016**, outperforming the individual models.
  - The final predictions were saved in `submission.csv` following the required submission format.

## Conclusion

The stacking ensemble of hyperparameter-tuned XGBoost and CatBoost models, with a RidgeCV meta-model, provided the best performance for predicting future blood glucose levels. By combining models that capture different aspects of the data patterns and using a meta-model to optimally integrate their predictions, the ensemble approach enhanced the robustness and accuracy of the results.

**Key Takeaways**:

- **Data Preprocessing**: Proper handling of missing values and feature scaling is crucial for model performance.
- **Feature Engineering**: Incorporating time-based features, moving averages, and interaction terms significantly improves predictive capabilities.
- **Model Ensemble**: Using a stacking ensemble allows for learning the best way to combine different models, leading to better performance than relying on a single model or simple averaging.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{brist1d,
    author = {Sam Gordon James and Miranda Elaine Glynis Armstrong and Aisling Ann O'Kane and Harry Emerson and Zahraa S. Abdallah},
    title = {BrisT1D Blood Glucose Prediction Competition},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/brist1d}},
    note = {Kaggle}
}
```

## Acknowledgements

- **BrisT1D Competition Organizers**: For providing the dataset and challenge.
- **Kaggle Community**: For resources and discussions that aided in developing the solution.
- **Machine Learning Libraries**: [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/).

---

**Disclaimer**: This project was developed for educational purposes as part of the BrisT1D Kaggle competition. The dataset and specific model implementations are subject to the competition's rules and guidelines.

## Contact

**Alvi Rownok**  
Email: [alvi2241998@gmail.com](mailto:alvi2241998@gmail.com)  
LinkedIn: [https://www.linkedin.com/in/alvi-rownok/](https://www.linkedin.com/in/alvi-rownok/)