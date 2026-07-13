California Housing Price Predictor (Linear Regression)



An introductory Machine Learning project implementing the complete ML lifecycle—from exploratory data analysis (EDA) to model evaluation—using the California Housing Dataset. 



\##  Project Overview

The goal of this project is to build a baseline Supervised Learning model utilizing Linear Regression to predict regional median house values (`MedHouseVal`) in California based on demographic and geographic features.



\## Tech Stack \& Libraries

&#x20;Language Python

&#x20;Environment Virtualenv \& Jupyter Notebook

&#x20;Libraries Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn



\## Key Results \& Performance

An 8020 traintest split was used to evaluate the model's predictive capabilities

&#x20;Mean Absolute Error (MAE) 0.533 (\~$53,300)

&#x20;Root Mean Squared Error (RMSE) 0.745

&#x20;R² Score 0.576 (The model explains roughly 57.6% of the variance in housing prices)



\### Major Insights from EDA

&#x20;Median Income (`MedInc`) proved to be the strongest linear predictor of housing prices with a correlation coefficient of \~0.69.

&#x20;The dataset features a hard ceiling cap at `MedHouseVal = 5.0` ($500,000), which introduces systematic under-prediction for high-value homes.



\## Future Scope for Improvement

1\. Non-Linear Models Transition from a baseline Linear Regression model to ensemble tree methods like Random Forest or XGBoost.

2\. Feature Engineering Create interaction attributes such as `RoomsPerPerson` to capture population density complexities.

3\. Outlier Mitigation Filter out or handle the capped premium properties ($500k ceiling) to reduce residual skewness.



\## Repository Contents

&#x20;`task1\_ml\_linear\_regression.ipynb` Complete Jupyter notebook containing the pipeline code and visual plots.

&#x20;`housing\_report.pdf` A short summary slide deck detailing the final findings.

