import json
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 1838)

with open('D:\Thesis\Thesis_coding\First_log_file\log0', 'r', encoding = 'utf-8') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

df = df.sort_values(by=['client', 'userid'])

df = df.reset_index()

df.drop(columns = 'index', inplace = True)

df['client'].unique()

df = df.applymap(lambda x: {} if isinstance(x, float) and np.isnan(x) else x)

def extract_help_count(events):
    help_count = 0
    for event in events:
        if event['event'] == 'HELP':
            value = event.get('value', 0)   # 0 is the default value returned if no key value exists
            if isinstance(value, (int, float)):
                help_count = int(value) + 1
            elif isinstance(value, str) and value.isdigit():   #isdigit returns True if the string is a number. ex: '56'(string) - True
                help_count = int(value) + 1
    return help_count

# Apply the function to extract help count for each row
df['help_count'] = df['events'].apply(lambda x: extract_help_count(x) if isinstance(x, list) else 0)

from datetime import datetime

def calculate_time_difference(df):
    start_times = {}
    time_differences = []

    for index, row in df.iterrows():
        if row['type'] == 'START':
            start_times[row['seed']] = datetime.strptime(row['_meta']['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
        elif row['type'] == 'FINISH':
            start_time = start_times.get(row['seed'])
            if start_time is not None:
                finish_time = datetime.strptime(row['_meta']['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
                time_difference = (finish_time - start_time).total_seconds()
                time_differences.append(time_difference)
            else:
                # If there is no corresponding start time, append None
                time_differences.append(None)

    # Pad with None values to match the length of the DataFrame
    num_pad = len(df) - len(time_differences)
    time_differences.extend([None] * num_pad)

    return time_differences

time_diffs = calculate_time_difference(df)

finish_counter = 0  # Counter to track the number of 'FINISH' events encountered

for index, time_diff in enumerate(time_diffs):
    if df.loc[index, 'type'] == 'FINISH':
        df.loc[index, 'time_taken_seconds'] = time_diffs[finish_counter]
        finish_counter += 1  # Move to the next time difference value

df['ageGroup'] = [row[0]['value']['ageGroup'] if isinstance(row, list) and len(row)>=3 and row and isinstance(row[0], dict) and 'value' in row[0] and isinstance(row[0]['value'], dict) and 'ageGroup' in row[0]['value'] else None for row in df['events']]
df['course'] = [row[1]['value']['course'] if isinstance(row, list) and row and len(row)>=3 and isinstance(row[1], dict) and 'value' in row[1] and isinstance(row[1]['value'], dict) and 'course' in row[1]['value'] else None for row in df['events']]
df['gender'] = [row[2]['value']['gender'] if isinstance(row, list) and row and len(row)>=3 and isinstance(row[2], dict) and 'value' in row[2] and isinstance(row[2]['value'], dict) and 'gender' in row[2]['value'] else None for row in df['events']]

df['ageGroup_mapped'] = df['ageGroup'].map({"0": "18-20", "1": "21-28", "2":"28+"})
df['course_mapped'] = df['course'].map({"0": "MME", "1": "info", "2":"andere"})
df['gender_mapped'] = df['gender'].map({"0": "manlich", "1": "weiblich", "2":"divers"})

def is_solved_extracted(rows):
        if isinstance(rows, dict) and 'isSolved' in rows:
            return rows['isSolved']
        else:
            None
            
df['is_Solved'] = df['results'].apply(is_solved_extracted)

df_mapped = df.drop(["ageGroup", "course", "gender"], axis = 1)

print(df_mapped)

df_mapped['client'] = df_mapped['client'].astype(str)

missing_clients = df_mapped.groupby('client').filter(lambda x: x[['ageGroup_mapped', 'course_mapped', 'gender_mapped']].isnull().all().all() or (x[['ageGroup_mapped', 'course_mapped', 'gender_mapped']] == '').all().all())

unique_missing_clients = missing_clients['client'].unique()

print("Clients with missing information:")
print(unique_missing_clients)

df_missing = df_mapped[df_mapped[['ageGroup_mapped', 'course_mapped', 'gender_mapped']].notnull().any(axis=1)]

print(df_missing['client'].unique())

df_data = df_mapped[df_mapped[['ageGroup_mapped', 'course_mapped', 'gender_mapped']].notnull().any(axis=1)]

df_data_unique = df_data.drop_duplicates(subset = 'client', keep = 'last') # first - drops the first duplicates

print(df_data_unique)

missing_info = df_mapped.groupby('client').filter(lambda x: x[['ageGroup_mapped', 'course_mapped', 'gender_mapped']].isnull().all().any() or 
                                                    (x[['ageGroup_mapped', 'course_mapped', 'gender_mapped']] == '').all().all())

unique_missing_clients = missing_info['client'].unique()

df_cleaned = df_mapped[~df_mapped['client'].isin(unique_missing_clients)]

df_cleaned = df_cleaned[~df_cleaned['id'].isin([''])]

df_cleaned['is_Solved'] = df_cleaned['is_Solved'].apply(lambda x: True if x == True else False if x == False else False)

df_cleaned['is_Solved'] = df_cleaned['is_Solved'].astype(int)

df_cleaned = df_cleaned.reset_index()

df_cleaned.drop(columns = 'index', inplace = True)

columns = ['ageGroup_mapped', 'course_mapped', 'gender_mapped']

for v in columns:
    # Find the first non-NaN entry
    non_na_indices = df_cleaned[pd.notna(df_cleaned[v])].index
    if len(non_na_indices) == 0:
        continue
    
    start_index = non_na_indices[0]

    # Initialize the group value
    grp_value = df_cleaned.at[start_index, v]

    for i in range(start_index, len(df_cleaned)):
        if pd.notna(df_cleaned.at[i, v]):
            # Update the group value if a new entry is found
            grp_value = df_cleaned.at[i, v]
        else:
            # Populate the column with the current group value
            df_cleaned.at[i, v] = grp_value

df_cleaned = df_cleaned[~df_cleaned['id'].isin(['courseSelectionExercise_de', 'surveystats'])]

df_cleaned = df_cleaned.reset_index()

df_cleaned.drop(columns = ['index'], inplace = True)

df_cleaned = df_cleaned[df_cleaned['results'] != {}].reset_index()

df_cleaned.drop(columns = ['index'], inplace = True)

print(df_cleaned)

grouped = df_cleaned.groupby('client')

# Apply 'describe' function to each group and store the results in a dictionary
describe_results = {client: group[['time_taken_seconds', 'help_count']].describe(include='all') for client, group in grouped}

# Print the results
for client, description in describe_results.items():
    client_info = grouped.get_group(client).iloc[0][['ageGroup_mapped', 'course_mapped', 'gender_mapped']]
    print(f"Client: {client}")
    print(f"Age Group: {client_info['ageGroup_mapped']}, Course: {client_info['course_mapped']}, Gender: {client_info['gender_mapped']}")
    print(description)
    print("\n")

grouped = df_cleaned.groupby(['client', 'ageGroup_mapped', 'gender_mapped', 'course_mapped'])

# Calculate and visualize correlation matrix for each client
# for group, data in grouped:
#     client, age_group, gender, course = group
#     print(f"Correlation matrix for Client: {client}")
    
#     # Selecting only numeric columns for correlation calculation
#     numeric_data = data.select_dtypes(include=['number'])
    
#     if not numeric_data.empty:
#         corr_matrix = numeric_data.corr()
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#         plt.title(f"Correlation Matrix for Client: {client}, Age Group: {age_group}, Gender: {gender}")
#         plt.show()
#     else:
#         print(f"No numeric data available for Client: {client}")

from scipy.stats import zscore

grouped = df_cleaned.groupby('client')

df_cleaned['time_taken_seconds'].fillna(df_cleaned['time_taken_seconds'].mean(), inplace=True)

solved_data = df_cleaned[df_cleaned['is_Solved'] == 1]

# Calculate z-scores for time_taken_seconds and help_count in the filtered data
solved_data['z_time_taken_seconds'] = zscore(solved_data['time_taken_seconds'])
solved_data['z_help_count'] = zscore(solved_data['help_count'])

df_cleaned.loc[solved_data.index, 'z_time_taken_seconds'] = solved_data['z_time_taken_seconds']
df_cleaned.loc[solved_data.index, 'z_help_count'] = solved_data['z_help_count']

# Initialize an empty DataFrame to store results
results = pd.DataFrame()

# Calculate aggregates for each client
for client, data in grouped:
    agg_data = data.groupby(['ageGroup_mapped', 'gender_mapped', 'course_mapped']).agg(
        mean_time_taken=('time_taken_seconds', 'mean'),
        std_time_taken=('time_taken_seconds', 'std'),
        mean_help_count=('help_count', 'mean'),
        std_help_count=('help_count', 'std'),
        count_solved=('is_Solved', 'sum'),
        count_mean=('is_Solved', 'mean'),
        total_tasks=('is_Solved', 'count'),
        mean_z_time_taken=('z_time_taken_seconds', 'mean'),
        mean_z_help_count=('z_help_count', 'mean')
    ).reset_index()
    
    agg_data['client'] = client
    results = pd.concat([results, agg_data], ignore_index=True)
    
# Display results
print(results)

#Total task counts for each group

grouped_age = results.groupby('ageGroup_mapped')

age_df = pd.DataFrame()

for ageGroup_mapped, data in grouped_age:
    agg_data = data.groupby(['ageGroup_mapped']).agg(
    Tasks_count = ('total_tasks', 'sum')
    )
    
    agg_data['ageGroup_mapped'] = ageGroup_mapped
    age_df = pd.concat([age_df, agg_data], ignore_index = True)    
    
Total_tasks_18_20 = age_df.loc[age_df['ageGroup_mapped'] == '18-20', 'Tasks_count'].values[0]
Total_tasks_21_28 = age_df.loc[age_df['ageGroup_mapped'] == '21-28', 'Tasks_count'].values[0]


plt.figure(figsize=(12, 8))
sns.barplot(x='ageGroup_mapped', y='Tasks_count', data=age_df)
plt.title('Total_tasks_attempted by each age_group')
plt.xlabel('Age Group')
plt.ylabel('Total_tasks_attempted')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = Total_tasks_18_20, Total_tasks_21_28  # heights of the two bars

plt.text(x1, y1 + 1, f'{Total_tasks_18_20:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2 + 1, f'{Total_tasks_21_28:.2f}', ha='right', va='bottom', color='orange')
plt.show()


grouped_gender = results.groupby('gender_mapped')

gender_df = pd.DataFrame()

for gender_mapped, data in grouped_gender:
    agg_data = data.groupby(['gender_mapped']).agg(
    Tasks_count = ('total_tasks', 'sum')
    )
    
    agg_data['gender_mapped'] = gender_mapped
    gender_df = pd.concat([gender_df, agg_data], ignore_index = True) 
    
Total_tasks_male = gender_df.loc[gender_df['gender_mapped'] == 'manlich', 'Tasks_count'].values[0]
Total_tasks_female = gender_df.loc[gender_df['gender_mapped'] == 'weiblich', 'Tasks_count'].values[0]

plt.figure(figsize=(12, 8))
sns.barplot(x='gender_mapped', y='Tasks_count', data=gender_df)
plt.title('Total_tasks_attempted by each gender')
plt.xlabel('gender')
plt.ylabel('Total_tasks_attempted')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = Total_tasks_male, Total_tasks_female  # heights of the two bars

plt.text(x1, y1 + 1, f'{Total_tasks_male:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2 + 1, f'{Total_tasks_female:.2f}', ha='right', va='bottom', color='orange')
plt.show()

#Average time taken by ech age group and gender

grouped_solved = solved_data.groupby('client')

results_solved = pd.DataFrame()

for client, data in grouped_solved:
    agg_data = data.groupby(['ageGroup_mapped', 'gender_mapped', 'course_mapped']).agg(
        mean_time_taken=('time_taken_seconds', 'mean'),
        mean_help_count=('help_count', 'mean'),
        mean_z_time_taken=('z_time_taken_seconds', 'mean'),
        mean_z_help_count=('z_help_count', 'mean')
    ).reset_index()
    
    agg_data['client'] = client
    results_solved = pd.concat([results_solved, agg_data], ignore_index=True)
    

# Calculate means for each age group
mean_time_18_20 = results_solved[results_solved['ageGroup_mapped'] == '18-20']['mean_time_taken'].mean()
mean_time_21_28 = results_solved[results_solved['ageGroup_mapped'] == '21-28']['mean_time_taken'].mean()

# Compute the percentage difference
percentage_diff = ((mean_time_18_20 - mean_time_21_28) / mean_time_21_28) * 100
    
# Bar plot for time taken by age group and gender
plt.figure(figsize=(12, 8))
sns.barplot(x='ageGroup_mapped', y='mean_time_taken', data=results_solved)
plt.title('Time Taken by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Time Taken (seconds)')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = mean_time_18_20, mean_time_21_28  # heights of the two bars

plt.text(x1, y1 + 1, f'{mean_time_18_20:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2 + 1, f'{mean_time_21_28:.2f}', ha='right', va='bottom', color='orange')
plt.text((x1 + x2) / 2, max(y1, y2) + 3, f'{percentage_diff:.2f}% difference', ha='center', va='bottom', color='black', fontsize = 30)
plt.show()

# Calculate means for each gender group
mean_time_male = results_solved[results_solved['gender_mapped'] == 'weiblich']['mean_time_taken'].mean()
mean_time_female = results_solved[results_solved['gender_mapped'] == 'manlich']['mean_time_taken'].mean()

# Compute the percentage difference
percentage_diff_gender = ((mean_time_male - mean_time_female) / mean_time_female) * 100

plt.figure(figsize=(12, 8))
sns.barplot(x='gender_mapped', y='mean_time_taken', data=results_solved)
plt.title('Time Taken by Gender')
plt.xlabel('Gender')
plt.ylabel('Mean Time Taken (seconds)')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = mean_time_male, mean_time_female  # heights of the two bars

plt.text(x1, y1 + 1, f'{mean_time_male:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2 + 1, f'{mean_time_female:.2f}', ha='right', va='bottom', color='orange')
plt.text((x1 + x2) / 2, max(y1, y2) + 2, f'{percentage_diff_gender:.2f}% difference', ha='center', va='bottom', color='black', fontsize = 30)
plt.show()

#Average help taken by each group

# Calculate means for each age group
mean_help_18_20 = results_solved[results_solved['ageGroup_mapped'] == '18-20']['mean_help_count'].mean()
mean_help_21_28 = results_solved[results_solved['ageGroup_mapped'] == '21-28']['mean_help_count'].mean()

# Compute the percentage difference
percentage_diff = ((mean_help_18_20 - mean_help_21_28) / mean_help_21_28) * 100
    
# Bar plot for time taken by age group and gender
plt.figure(figsize=(12, 8))
sns.barplot(x='ageGroup_mapped', y='mean_help_count', data=results_solved)
plt.title('Help Taken by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Mean Help count (per question)')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = mean_help_18_20, mean_help_21_28  # heights of the two bars

plt.text(x1, y1, f'{mean_help_18_20:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2, f'{mean_help_21_28:.2f}', ha='right', va='bottom', color='orange')
plt.text((x1 + x2) / 2, max(y1, y2) + 0.3, f'{percentage_diff:.2f}% difference', ha='center', va='bottom', color='black', fontsize = 30)
plt.show()

# Calculate means for each gender group
mean_help_female = results_solved[results_solved['gender_mapped'] == 'weiblich']['mean_help_count'].mean()
mean_help_male = results_solved[results_solved['gender_mapped'] == 'manlich']['mean_help_count'].mean()

# Compute the percentage difference
percentage_diff_gender = ((mean_help_female - mean_help_male) / mean_help_female) * 100

plt.figure(figsize=(12, 8))
sns.barplot(x='gender_mapped', y='mean_help_count', data=results_solved)
plt.title('Help Taken by Gender')
plt.xlabel('Gender')
plt.ylabel('Mean Help count (per question)')
plt.xticks(rotation=45)

# Annotate the plot with percentage difference
x1, x2 = 0, 1  # positions of the two bars
y1, y2 = mean_help_female, mean_help_male  # heights of the two bars

plt.text(x1, y1, f'{mean_help_female:.2f}', ha='right', va='bottom', color='blue')
plt.text(x2, y2, f'{mean_help_male:.2f}', ha='right', va='bottom', color='orange')
plt.text((x1 + x2) / 2, max(y1, y2) + 0.2, f'{percentage_diff_gender:.2f}% difference', ha='center', va='bottom', color='black', fontsize = 30)
plt.show()

#Hypotheses answering

# Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import statsmodels.api as sm

df_encoded = pd.get_dummies(df_cleaned, columns=['ageGroup_mapped', 'gender_mapped'], drop_first=True)

df_encoded.rename(columns={'ageGroup_mapped_21-28': 'ageGroup_mapped', 'gender_mapped_weiblich': 'gender_mapped'}, inplace=True)

# Define features and target
# X = df_encoded[[col for col in df_encoded.columns if col.startswith('ageGroup_mapped') or col.startswith('gender_mapped')]]
X = df_encoded[['ageGroup_mapped', 'gender_mapped']]
y = df_encoded['is_Solved']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_train = X_train.astype(int)

y_train = y_train.astype(int)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(f"confusion matrix: {cm}")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Fit the logistic regression model using statsmodels
X_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_sm)
result = logit_model.fit()

# Print the summary of the model
print(result.summary())

# Extract coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Form the logistic regression equation
equation = f"logit(P) = {intercept} + ({coefficients[0]} * ageGroup) + ({coefficients[1]} * gender)"
print("Logistic Regression Equation:")
print(equation)

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

X = df_encoded[['ageGroup_mapped', 'gender_mapped']]
y = df_encoded['is_Solved']

# Create and train the Random Forest model
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_dt = grid_search.best_estimator_

# Make predictions
y_pred = best_dt.predict(X_test)
y_pred_prob = best_dt.predict_proba(X_test)[:, 1]

# Confusion matrix and classification report
accuracy = accuracy_score(y_test, y_pred)
cm_rf = confusion_matrix(y_test, y_pred)
cr_rf = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(f"confusion matrix: {cm_rf}")
print(cr_rf)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Visualize one of the trees in the Random Forest
# plt.figure(figsize=(20, 10))
# tree_num = 0  # Index of the tree to plot
# plot_tree(rf_model.estimators_[tree_num], feature_names=X.columns, filled=True, rounded=True, class_names=True)
# plt.show()

plt.figure(figsize=(20,10))
tree.plot_tree(best_dt, feature_names=['ageGroup_mapped', 'gender_mapped'], class_names=['Not Solved', 'Solved'], filled=True, label='none', impurity=False, node_ids=False, proportion=True, rounded=True, precision=2, fontsize=26)
plt.show()

# Get feature importances
importances = best_dt.feature_importances_
features = X.columns

for feature, importance in zip(features, importances):
    print(f'{feature}: {importance}')

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

print(feature_importance_df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Encode categorical variables
df_encoded = pd.get_dummies(df_cleaned, columns=['ageGroup_mapped', 'gender_mapped'], drop_first=True)

df_encoded.rename(columns={'ageGroup_mapped_21-28': 'ageGroup_mapped', 'gender_mapped_weiblich': 'gender_mapped'}, inplace=True)

# Define features and target
X = df_encoded[['ageGroup_mapped', 'gender_mapped']]
y = df_encoded['is_Solved']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Extracting coefficients for linear SVM
coefficients = pd.DataFrame(svm_model.coef_, columns=['ageGroup_mapped', 'gender_mapped'])
print(coefficients)

# Assuming a binary classification with two features
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Age Group')
    plt.ylabel('Gender')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X_test, y_test, svm_model)

#gridsearch cv

#Making suggestions

grouped_unsolved = df_cleaned.groupby('id')

results_unsolved = pd.DataFrame()

for id, data in grouped_unsolved:
    agg_data = data.groupby(['ageGroup_mapped', 'gender_mapped']).agg(
        total_tasks=('is_Solved', 'count'),
        True_count=('is_Solved', 'sum'),
        False_count=('is_Solved', lambda x: (x == 0).sum()),
        False_count_mean=('is_Solved', lambda x: (x == 0).mean())
    ).reset_index()
    
    agg_data['id'] = id
    results_unsolved = pd.concat([results_unsolved, agg_data], ignore_index=True)

prescriptive = df_cleaned.groupby('id')

results_sug = pd.DataFrame()

for id, data in prescriptive:    #whatever parameter is used in groupby should be used as the first parameter in the for loop here
    
    agg_data = data.groupby(['ageGroup_mapped', 'gender_mapped']).agg(
        total_tasks=('is_Solved', 'count'),
        True_count=('is_Solved', 'sum'),
        False_count=('is_Solved', lambda x: (x == 0).sum()),
        False_count_mean=('is_Solved', lambda x: (x == 0).mean())
    ).reset_index()

    solved_data = data[data['is_Solved'] == 1]
    
    # Aggregate mean_time_taken and mean_help_count only for solved_data
    solved_agg_data = solved_data.groupby(['ageGroup_mapped', 'gender_mapped']).agg(
        mean_time_taken=('time_taken_seconds', 'mean'),
        mean_help_count=('help_count', 'mean')
    ).reset_index()

    # Merge the two aggregated DataFrames on 'ageGroup_mapped' and 'gender_mapped'
    merged_agg_data = pd.merge(agg_data, solved_agg_data, on=['ageGroup_mapped', 'gender_mapped'])

    # Add the 'id' column
    merged_agg_data['id'] = id

    # Concatenate the merged data to the results DataFrame
    results_sug = pd.concat([results_sug, merged_agg_data], ignore_index=True)

threshold = pd.DataFrame()

threshold = results_sug.groupby('id').agg(
    mean_false_count=('False_count', 'mean'),
    mean_time_taken=('mean_time_taken', 'mean'),
    mean_help_count=('mean_help_count', 'mean')
).reset_index()
    
merged_df = pd.merge(results_sug, threshold, on='id', suffixes=('', '_mean'))

# Calculate the thresholds
false_count_threshold = merged_df['mean_false_count'] 
time_taken_threshold = merged_df['mean_time_taken_mean']
help_count_threshold = merged_df['mean_help_count_mean']

# Apply the conditions
merged_df['suggestion_needed'] = (
    (merged_df['False_count'] > false_count_threshold) |
    (merged_df['mean_time_taken'] > time_taken_threshold) |
    (merged_df['mean_help_count'] > help_count_threshold)
).astype(int)  

df_suggestions = merged_df[merged_df['suggestion_needed'] == 1]

grouped = df_suggestions.groupby(['ageGroup_mapped', 'gender_mapped', 'id']).size().reset_index(name='counts')

# Create a pivot table for the bar plot
pivot = grouped.pivot_table(index=['ageGroup_mapped', 'gender_mapped'], columns='id', values='counts', fill_value=0)

# Plotting
pivot.plot(kind='bar', stacked=True, figsize=(14, 8))

# Add title and labels
plt.title('Math Topics Needing Help by Age Group and Gender')
plt.xlabel('Age Group and Gender')
plt.ylabel('Count of Math Topics Needing Help')

# Show the plot
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend(title='Math Topics')
plt.show()