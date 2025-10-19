# Admission Chance Prediction Project
# ----------------------------------
# Single-file Jupyter/Script style project that:
#  - Loads the Admission Chance dataset
#  - Performs EDA and basic preprocessing
#  - Trains multiple models (Linear Regression, Random Forest, Gradient Boosting)
#  - Runs cross-validation and hyperparameter search for the best model
#  - Evaluates models with MAE, MSE, RMSE, MAPE and R^2
#  - Saves the best model to disk
#  - Provides a small Streamlit app example at the bottom for quick UI

# NOTE: Run this file in a Jupyter notebook or as a Python script. If running as a script,
# comment out the plotting cells or use a backend that supports plotting.

# Requirements:
# pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit

# ----------------------------------
# Step 0: Imports
# ----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 2529

# ----------------------------------
# Step 1: Load data
# ----------------------------------
url = 'https://github.com/ybifoundation/Dataset/raw/main/Admission%20Chance.csv'
admission = pd.read_csv(url)

# Quick look
print('\nData shape:', admission.shape)
print('\nColumns:\n', admission.columns.tolist())

# The dataset has a column named 'Chance of Admit ' with a trailing space. We'll clean column names.
admission.columns = [c.strip() for c in admission.columns]

# Drop Serial No because it is just an index
if 'Serial No' in admission.columns:
    admission = admission.drop(columns=['Serial No'])

print('\nCleaned columns:\n', admission.columns.tolist())
print('\nFirst 5 rows:')
print(admission.head())

# ----------------------------------
# Step 2: Basic EDA
# ----------------------------------
print('\nDataset info:')
print(admission.info())

print('\nSummary statistics:')
print(admission.describe())

# Check missing values
print('\nMissing values per column:\n', admission.isna().sum())

# Explore correlations
corr = admission.corr()
print('\nCorrelation with target:\n', corr['Chance of Admit'].sort_values(ascending=False))

# Plot pairwise relationships for a subset - uncomment in notebooks
try:
    sns.pairplot(admission[['GRE Score','TOEFL Score','CGPA','LOR','SOP','Chance of Admit']])
    plt.suptitle('Pairwise plots (subset)', y=1.02)
    plt.show()
except Exception:
    pass

# Heatmap of correlations
try:
    plt.figure(figsize=(9,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation matrix')
    plt.show()
except Exception:
    pass

# ----------------------------------
# Step 3: Prepare X and y
# ----------------------------------
X = admission.drop(columns=['Chance of Admit'])
y = admission['Chance of Admit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=RANDOM_STATE)
print('\nTrain/Test shapes:', X_train.shape, X_test.shape)

# ----------------------------------
# Step 4: Baseline model - Linear Regression
# ----------------------------------
# We'll standardize features for linear models
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

pipeline_lr.fit(X_train, y_train)

y_pred_lr = pipeline_lr.predict(X_test)

def evaluate(y_true, y_pred, prefix=''):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{prefix}MAE: {mae:.4f}")
    print(f"{prefix}MSE: {mse:.6f}")
    print(f"{prefix}RMSE: {rmse:.4f}")
    print(f"{prefix}MAPE: {mape:.4f}")
    print(f"{prefix}R^2: {r2:.4f}")
    return {'mae':mae,'mse':mse,'rmse':rmse,'mape':mape,'r2':r2}

print('\nLinear Regression performance:')
metrics_lr = evaluate(y_test, y_pred_lr, prefix='[LR] ')

# Coefficients interpretation (after scaling):
try:
    # access scaler & linear model
    scaler = pipeline_lr.named_steps['scaler']
    lr = pipeline_lr.named_steps['lr']
    coef = lr.coef_
    features = X.columns.tolist()
    coef_df = pd.DataFrame({'feature':features, 'coef':coef})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
    print('\nLinear model coefficients:')
    print(coef_df[['feature','coef']])
except Exception:
    pass

# ----------------------------------
# Step 5: Tree-based models (Random Forest and Gradient Boosting)
# ----------------------------------
rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
gb = GradientBoostingRegressor(random_state=RANDOM_STATE)

# We'll do a small grid search for RF
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, None],
    'min_samples_split': [2, 5]
}

rf_gs = GridSearchCV(rf, rf_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
rf_gs.fit(X_train, y_train)

print('\nBest Random Forest params:', rf_gs.best_params_)
rf_best = rf_gs.best_estimator_

# Fit Gradient Boosting with a small grid
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

gb_gs = GridSearchCV(gb, gb_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
gb_gs.fit(X_train, y_train)

print('\nBest Gradient Boosting params:', gb_gs.best_params_)
gb_best = gb_gs.best_estimator_

# Evaluate RF and GB on test set
y_pred_rf = rf_best.predict(X_test)
y_pred_gb = gb_best.predict(X_test)

print('\nRandom Forest performance:')
metrics_rf = evaluate(y_test, y_pred_rf, prefix='[RF] ')
print('\nGradient Boosting performance:')
metrics_gb = evaluate(y_test, y_pred_gb, prefix='[GB] ')

# ----------------------------------
# Step 6: Compare models and select best by MAE (or any other metric)
# ----------------------------------
results = pd.DataFrame([
    {'model':'LinearRegression','mae':metrics_lr['mae'],'rmse':metrics_lr['rmse'],'r2':metrics_lr['r2']},
    {'model':'RandomForest',   'mae':metrics_rf['mae'], 'rmse':metrics_rf['rmse'], 'r2':metrics_rf['r2']},
    {'model':'GradientBoost',  'mae':metrics_gb['mae'], 'rmse':metrics_gb['rmse'], 'r2':metrics_gb['r2']}
])
print('\nModel comparison:')
print(results.sort_values('mae'))

# Choose best model (here choose the model with lowest MAE)
best_name = results.sort_values('mae').iloc[0]['model']
print('\nBest model according to MAE:', best_name)

if best_name == 'RandomForest':
    best_model = rf_best
elif best_name == 'GradientBoost':
    best_model = gb_best
else:
    best_model = pipeline_lr

# ----------------------------------
# Step 7: Save model
# ----------------------------------
model_filename = 'admission_best_model.joblib'
joblib.dump(best_model, model_filename)
print(f"\nSaved best model to: {model_filename}")

# Save a simple scaler (for linear model pipeline) if used
if best_name == 'LinearRegression':
    # pipeline already saved
    pass

# ----------------------------------
# Step 8: Feature importance (for tree models)
# ----------------------------------
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    print('\nFeature importances:')
    print(feat_imp)
    try:
        plt.figure(figsize=(8,5))
        sns.barplot(x='importance', y='feature', data=feat_imp)
        plt.title('Feature importances')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

# ----------------------------------
# Step 9: Residual diagnostics for best model
# ----------------------------------
if isinstance(best_model, Pipeline):
    y_pred_all = best_model.predict(X_test)
else:
    y_pred_all = best_model.predict(X_test)

residuals = y_test - y_pred_all
try:
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred_all, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.show()
except Exception:
    pass

# ----------------------------------
# Step 10: Helper function to predict a single student
# ----------------------------------
feature_order = X.columns.tolist()
print('\nFeature order used for prediction:', feature_order)

def predict_single(input_dict, model=best_model):
    """
    input_dict: dictionary with keys matching the dataset columns, e.g.
      {
        'GRE Score': 330,
        'TOEFL Score': 115,
        'University Rating': 5,
        'SOP': 4.5,
        'LOR': 4.5,
        'CGPA': 9.0,
        'Research': 1
      }
    Returns predicted Chance of Admit (float)
    """
    x = pd.DataFrame([input_dict], columns=feature_order)
    pred = model.predict(x)[0]
    return float(pred)

# Example prediction
example = {
    'GRE Score': 330,
    'TOEFL Score': 115,
    'University Rating': 5,
    'SOP': 4.5,
    'LOR': 4.5,
    'CGPA': 9.0,
    'Research': 1
}
print('\nExample predicted chance of admit:', round(predict_single(example), 4))

# ----------------------------------
# OPTIONAL: Minimal Streamlit UI
# ----------------------------------
# Save this block into a separate file (app.py) if you want to run the Streamlit app:
streamlit_app = '''
# admission_app.py
import streamlit as st
import joblib
import pandas as pd

st.title('Graduate Admission Chance Predictor')
model = joblib.load('admission_best_model.joblib')

# Input fields
GRE = st.slider('GRE Score', 290, 340, 320)
TOEFL = st.slider('TOEFL Score', 92, 120, 110)
Uni = st.selectbox('University Rating', [1,2,3,4,5], index=3)
SOP = st.slider('SOP (1-5)', 1.0, 5.0, 3.5)
LOR = st.slider('LOR (1-5)', 1.0, 5.0, 3.5)
CGPA = st.slider('CGPA (0-10)', 6.8, 9.92, 8.6)
Research = st.selectbox('Research (0 or 1)', [0,1], index=1)

if st.button('Predict'):
    x = pd.DataFrame([{ 'GRE Score':GRE, 'TOEFL Score':TOEFL, 'University Rating':Uni, 'SOP':SOP, 'LOR':LOR, 'CGPA':CGPA, 'Research':Research }])
    pred = model.predict(x)[0]
    st.write('Predicted Chance of Admit:', round(float(pred), 4))
'''

print('\n\nIf you want a Streamlit app, write the content of variable `streamlit_app` to a file named admission_app.py and run:')
print('  streamlit run admission_app.py')

# Write the streamlit app string to disk as an example file
with open('admission_app.py', 'w') as f:
    f.write(streamlit_app)
print('\nWrote example Streamlit app to admission_app.py')

# ----------------------------------
# End of project file
# ----------------------------------
