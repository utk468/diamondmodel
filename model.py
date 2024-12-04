import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("diamonds.csv")
print(data.head())

print(data.shape)
#Steps involved in Data Preprocessing
# Data cleaning
# Identifying and removing outliers
# Encoding categorical variables

print(data.info())
#The first column seems to be just index
data = data.drop(["Unnamed"], axis=1)
print(data)


#Dropping dimentionless diamonds
data = data.drop(data[data["x"]==0].index)
data = data.drop(data[data["y"]==0].index)
data = data.drop(data[data["z"]==0].index)
print(data.shape)

 #finding outlier
def find_outliers_iqr(dataframe):
    outlier_indices = {}

    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        outlier_indices[column] = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)].index.tolist()

    return outlier_indices


outliers = find_outliers_iqr(data)


for column, indices in outliers.items():
    if indices:
        print(f"\nOutliers for column '{column}':")
        print(f"  Outlier indices: {indices}")
    else:
        print(f"\nNo outliers found for column '{column}'.")



#graph with outlier
sns.set(style="whitegrid")

melted_df = data.melt(var_name='Feature', value_name='Value', value_vars=data.select_dtypes(include=[np.number]).columns)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Feature', y='Value', data=melted_df)
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()




# remove outlier
def remove_outliers_iqr(dataframe):
    cleaned_df = dataframe.copy()
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]

    return cleaned_df


df_cleaned = remove_outliers_iqr(data)


print(f"Original dataset shape: {data.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")


#graph without outlier
sns.set(style="whitegrid")

melted_df = df_cleaned.melt(var_name='Feature', value_name='Value', value_vars=data.select_dtypes(include=[np.number]).columns)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Feature', y='Value', data=melted_df)
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()


# Get list of categorical variables
s = (data.dtypes =="object")
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)


# Make copy to avoid changing original data 
label_data = data.copy()


# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_data[col] = label_encoder.fit_transform(label_data[col])
print(label_data.head())

#correlation matrix
plt.figure(figsize = (15,15))
sns.heatmap(label_data.corr(),annot=True, cmap='Blues', fmt='.1f')
plt.show()



# Assigning the featurs as X and trarget as y
X= label_data.drop(["price"],axis =1)
y= label_data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=7)


# Building pipelins of standard scaler and model for varios regressors.
# 1. Consistency in Preprocessing
# 2. Avoiding Data Leakage
# 3. Simplifies Code and Reduces Errors
# 4. Hyperparameter Tuning
# 5. Different Models Require Different Preprocessing


# Some models require feature 
# scaling (like LinearRegression, 
# Support Vector Machines, 
# K-Nearest Neighbors), while 
# others may not (like Decision
#  Trees or Random Forests).
# By creating pipelines for 
# various regressors, you can
# handle the different preprocessing 
# needs required by each model
# automatically.


pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr_classifier",LinearRegression())])

pipeline_dt=Pipeline([("scalar2",StandardScaler()),
                     ("dt_classifier",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar3",StandardScaler()),
                     ("rf_classifier",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar4",StandardScaler()),
                     ("rf_classifier",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar5",StandardScaler()),
                     ("rf_classifier",XGBRegressor())])

pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)


cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))  
    # This prints the average cross-validation 
    # score (cv_score.mean()) for each pipeline.


# In the code 
# the goal is to evaluate the 
# performance of each model
# (defined in pipelines) using 
# cross-validation and a specific 
# scoring metric (Root Mean Squared
# Error, neg_root_mean_squared_error).     


# Root Mean Squared Error (RMSE)
# is a widely used evaluation metric
# for regression models.

# Negative RMSE (neg_root_mean_
# squared_error) is used because 
# cross_val_score returns a 
# score that should be maximized,
# and lower RMSE means better
# performance. So, the negative 
# sign is applied to convert
# it into a maximization problem
# (i.e., higher negative values are
# worse, and less negative is better).





# Testing the Model with the best 
# score on the test set

# In the above scores,
# XGBClassifier appears to be 
# the model with the best scoring
# on negative root mean square error

# Model prediction on test data
pred = pipeline_xgb.predict(X_test)
print("R^2:",r2_score(y_test, pred))
#A higher R² score indicates a better fit and more accurate predictions from the model


label_encoder_cut = LabelEncoder().fit(data['cut'])
label_encoder_color = LabelEncoder().fit(data['color'])
label_encoder_clarity = LabelEncoder().fit(data['clarity'])
#2,Very Good,D,SI2,60,56,17953,8.16,8.23,4.92
#17299
# Define default input values
carat = 2
cut = 'Very Good'          # Example cut
color = 'D'          # Example color
clarity = 'SI2'      # Example clarity
depth = 60      # Example depth
table = 56       # Example table
x = 8.16            # Example x dimension
y = 8.23           # Example y dimension
z = 4.92            # Example z dimension

# Encode categorical inputs
cut_encoded = label_encoder_cut.transform([cut])[0]
color_encoded = label_encoder_color.transform([color])[0]
clarity_encoded = label_encoder_clarity.transform([clarity])[0]

# Prepare input data in the same format as training data
input_data = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]])

# Predict using the trained pipeline_xgb
predicted_price = pipeline_xgb.predict(input_data)

print(f"The predicted price for the diamond is: ${predicted_price[0]:.2f}")


joblib.dump(label_encoder_cut, 'label_encoder_cut.pkl')
joblib.dump(label_encoder_color, 'label_encoder_color.pkl')
joblib.dump(label_encoder_clarity, 'label_encoder_clarity.pkl')

# Save the pipeline_xgb model
joblib.dump(pipeline_xgb, 'pipeline_xgb_model.pkl')

print("Model saved as 'pipeline_xgb_model.pkl'")



# 0.41,Good,G,VVS1,61,61,986,4.77,4.8,2.92
#1012
#0.24,Very Good,E,VVS2,63.9,53,485,3.93,3.96,2.52
#520
#2,Very Good,D,SI2,60,56,17953,8.16,8.23,4.92
#17299
#0.91,Premium,F,SI2,62.1,56,3096,6.26,6.21,3.87
#3038


