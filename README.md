# House Price Prediction Project

This project implements a machine learning pipeline to predict house prices based on socio-economic and demographic features.

## Dataset
The dataset is available on Kaggle: [USA House Prices](https://www.kaggle.com/datasets/tanyachawla412/house-prices).

## Key Findings & Results
The project utilized **Linear Regression** as the primary model. After refining the feature selection (removing 'Number of Bedrooms' due to low impact) and cleaning outliers using the IQR method, the model's performance improved.

### Performance Metrics (Refined & Cleaned Data)
- **R2 Score:** 0.8849 (approx. 88.5%)
- **MAE:** 93,429
- **Interpretation:** The refined model demonstrates high accuracy, capturing 88.5% of the price variance. On average, predictions deviate by approximately $93k from the actual values.

### Feature Importance (Coefficients)
The standardized coefficients indicate how much each feature influences the house price:
| Feature | Coefficient |
| :--- | :--- |
| Avg. Area Income | 225,485.25 |
| House Age | 156,336.91 |
| Area Population | 149,653.84 |
| Number of Rooms | 118,932.16 |

## Deployment
The model and the scaler are saved using **Joblib** for efficient serialization. This ensures that the preprocessing (scaling) and prediction logic remain consistent in a production environment.
- `model.pkl`: The trained Linear Regression model.
- `scaler.pkl`: The StandardScaler instance used during training.
