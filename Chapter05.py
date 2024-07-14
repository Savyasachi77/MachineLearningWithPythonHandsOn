import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor and fit it to the training data
tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg.fit(X_train, y_train)

# Evaluate the model on the test set
score = tree_reg.score(X_test, y_test)
print(f"R^2 Score: {score}")

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(tree_reg, feature_names=data.feature_names, filled=True, rounded=True)
plt.show()
import xgboost as xgb
from sklearn.metrics import mean_squared_error
# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the datasets into DMatrix, which is a data format used by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost parameters
params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}
num_rounds = 100

# Train the model
bst = xgb.train(params, dtrain, num_rounds)

# Make predictions and evaluate the model
predictions = bst.predict(dtest)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Feature importance
xgb.plot_importance(bst)
plt.show()
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg.fit(X_train, y_train)
tree_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

# XGBoost Regressor
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)
xgb_pred = bst.predict(dtest)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# Comparison
print("Decision Tree Regressor:")
print(f"Mean Squared Error: {tree_mse}")
print(f"R^2 Score: {tree_r2}\n")

print("XGBoost Regressor:")
print(f"Mean Squared Error: {xgb_mse}")
print(f"R^2 Score: {xgb_r2}")
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'max_depth': 3,  # Keeping it shallow for visualization purposes
    'eta': 0.1,
    'objective': 'reg:squarederror',
}
num_boost_round = 10  # Limiting the number of boosting rounds

bst = xgb.train(params, dtrain, num_boost_round)

# Draw the first tree
xgb.plot_tree(bst, num_trees=0)
plt.show()

# Save the tree as a file (optional)
# bst.dump_model('dump.raw.txt')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the datasets into DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)

# Define XGBoost parameters
params = {
    'max_depth': 3,  # Keeping it shallow for better visualization
    'eta': 0.1,
    'objective': 'reg:squarederror'
}
num_boost_round = 10

# Train the model
bst = xgb.train(params, dtrain, num_boost_round)

xgb.plot_tree(bst)

# Plotting the first tree
plt.figure(figsize=(80, 60))
#plt.subplot(2, 1, 1)  # Two plots in one column: this is the first
xgb.plot_tree(bst, num_trees=0, rankdir='LR')  # Tree 0
plt.title("First Tree")

# Plotting the second tree
#plt.subplot(2, 1, 2)  # Two plots in one column: this is the second
xgb.plot_tree(bst, num_trees=1, rankdir='LR')  # Tree 1
plt.title("Second Tree")
# Print feature names corresponding to f0, f1, etc.
print("Feature mapping:")
for i, feature in enumerate(feature_names):
    print(f"f{i}: {feature}")

plt.show()
print(feature_names)
data.feature_names
data.data
import pandas as pd
x = pd.DataFrame(data.data, columns=data.feature_names)
x.head()
data
data.DESCR
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Load the dataset
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedianHouseValue'] = california_housing.target

# Select a feature and split value
feature = 'MedInc'
split_value = df[feature].median()

# Split the dataset
left_group = df[df[feature] < split_value]
right_group = df[df[feature] >= split_value]

# Calculate variance
var_parent = df['MedianHouseValue'].var()
var_left = left_group['MedianHouseValue'].var()
var_right = right_group['MedianHouseValue'].var()

# Variance reduction
N_parent = len(df)
N_left = len(left_group)
N_right = len(right_group)
var_reduction = var_parent - ((N_left / N_parent) * var_left + (N_right / N_parent) * var_right)

# Visualizing the split
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(left_group['MedianHouseValue'], bins=30, alpha=0.7, label='Left Group')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Left Group after Split')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(right_group['MedianHouseValue'], bins=30, alpha=0.7, color='orange', label='Right Group')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Right Group after Split')
plt.legend()

plt.tight_layout()
plt.show()

print("Variance Parent:", var_parent)
print("Variance Left:", var_left)
print("Variance Right:", var_right)
print("Variance Reduction:", var_reduction)
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = california_housing.target

# Train a decision tree regressor
# Limiting the depth for visualization purposes
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(regressor, feature_names=california_housing.feature_names, filled=True)
plt.show()
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = (data.target >= 2).astype(int)  # 1 if 'High Cost', 0 otherwise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = tree_clf.predict(X_test)
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the entire decision tree
plt.figure(figsize=(20,10))
plot_tree(tree_clf, feature_names=data.feature_names, class_names=["Low Cost", "High Cost"], filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree for California Housing Dataset")
plt.show()
# Zooming into a specific part of the decision tree
plt.figure(figsize=(12,8))
plot_tree(tree_clf, 
feature_names=data.feature_names, 
class_names=["Low Cost", "High Cost"], 
filled=True, rounded=True, 
fontsize=12,
max_depth=2) # Limiting the depth for zooming in
plt.title("Zoomed-in View of Decision Tree (Depth 2)")
plt.show()
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming tree_reg is the trained Decision Tree Regressor
plt.figure(figsize=(20,10))
plot_tree(tree_reg, 
          feature_names=data.feature_names, 
          filled=True, rounded=True, 
          fontsize=12)
plt.title("Regression Tree Visualization")
plt.show()
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, scoring='neg_mean_squared_error'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

    # Calculate mean and standard deviation for training set scores
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.show()

# Example usage with a DecisionTreeRegressor
plot_learning_curve(DecisionTreeRegressor(max_depth=4), X, y)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_fitting_curves(X, y, max_depths, train_size):
    plt.figure(figsize=(10, 6))

    for max_depth in max_depths:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size)
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_error = mean_squared_error(y_train, y_train_pred)
        val_error = mean_squared_error(y_val, y_val_pred)

        plt.plot(max_depth, train_error, 'ro-', label='Training error' if max_depth == max_depths[0] else "")
        plt.plot(max_depth, val_error, 'bs-', label='Validation error' if max_depth == max_depths[0] else "")

    plt.xlabel('Model Complexity (Max Tree Depth)')
    plt.ylabel('Mean Squared Error')
    plt.title('Overfitting, Underfitting, and Just-Right Fitting in Regression Trees')
    plt.legend()
    plt.axvline(x=max_depths[1], color='gray', linestyle='--', label='Just-Right Fitting')
    plt.annotate('Underfitting', xy=(1, 0.5), xytext=(1, 0.5))
    plt.annotate('Overfitting', xy=(max(max_depths), 0.5), xytext=(max(max_depths), 0.5))
    plt.show()

# Example usage
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]  # Depths representing underfitting
train_size = 10000
plot_fitting_curves(X, y, max_depths, train_size)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Fetching the dataset
data = fetch_california_housing()
X, y = data.data, data.target

def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error', cv=5)

    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, val_scores_mean, label='Validation error')
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.ylim(0, 3)  # Adjust as needed
    plt.show()

# Plotting for different depths
for depth in [1, 3, 10]:  # Representing underfitting, just-right, and overfitting
    print(f"Learning curve for tree depth: {depth}")
    plot_learning_curves(DecisionTreeRegressor(max_depth=depth), X, y)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the California Housing dataset
# Assuming X_train, X_test, y_train, y_test are already defined

# Train a regression tree with no restrictions (prone to overfitting)
tree_reg_overfit = DecisionTreeRegressor(random_state=42)
tree_reg_overfit.fit(X_train, y_train)

# Predictions and MSE for training and test data
y_train_pred_overfit = tree_reg_overfit.predict(X_train)
y_test_pred_overfit = tree_reg_overfit.predict(X_test)
mse_train_overfit = mean_squared_error(y_train, y_train_pred_overfit)
mse_test_overfit = mean_squared_error(y_test, y_test_pred_overfit)

print(f"Training MSE: {mse_train_overfit}")
print(f"Test MSE: {mse_test_overfit}")

# Plot for training data
plt.scatter(y_train, y_train_pred_overfit, alpha=0.6)
plt.title("Predictions vs. Actual Values on Training Data")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

# Plot for test data
plt.scatter(y_test, y_test_pred_overfit, alpha=0.6)
plt.title("Predictions vs. Actual Values on Test Data")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, scoring='neg_mean_squared_error'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

    # Calculate mean and standard deviation for training set scores
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.show()

# Example usage with a DecisionTreeRegressor
plot_learning_curve(DecisionTreeRegressor(max_depth=4), X, y)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_california_housing(return_X_y=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, 
                          learning_rate = 0.15, max_depth = 5, alpha = 10, n_estimators = 20)

# Train the model
xg_reg.fit(X_train, y_train)

# Predict and evaluate
preds = xg_reg.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("MSE for XGBoost: ", mse)
from sklearn.tree import DecisionTreeRegressor

# Initialize Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(X_train, y_train)

# Predict and evaluate
tree_preds = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_preds)
print("MSE for Decision Tree: ", tree_mse)
plt.bar(['XGBoost', 'Decision Tree'], [mse, tree_mse], color=['blue', 'green'])
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison: XGBoost vs Decision Tree')
plt.show()
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_hyperparameter_effects(param_name, param_values):
    mses = []
    for value in param_values:
        params = {param_name: value, 'objective': 'reg:squarederror'}
        xg_reg = xgb.XGBRegressor(**params)
        xg_reg.fit(X_train, y_train)
        preds = xg_reg.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mses.append(mse)
    
    plt.plot(param_values, mses)
    plt.title(f'Effect of {param_name} on MSE')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error')
    plt.show()

# Example: Plotting the effect of 'max_depth'
plot_hyperparameter_effects('max_depth', range(1, 10))
# Plotting the effect of 'learning_rate'
plot_hyperparameter_effects('learning_rate', [0.01, 0.05, 0.1, 0.3, 0.5])
# Plotting the effect of 'n_estimators'
plot_hyperparameter_effects('n_estimators', [10, 50, 100, 200])
def plot_hyperparams_effect(X_train, y_train, X_test, y_test, hyperparams, param_values):
    train_errors, test_errors = [], []

    for value in param_values:
        params = {hyperparams: value, 'objective': 'reg:squarederror'}
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train, train_preds))
        test_errors.append(mean_squared_error(y_test, test_preds))

    plt.figure(figsize=(8, 4))
    plt.plot(param_values, train_errors, label='Training Error')
    plt.plot(param_values, test_errors, label='Testing Error')
    plt.title(f'Effect of {hyperparams} on Training and Testing Error')
    plt.xlabel(hyperparams)
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

# Plotting the effect of learning_rate
plot_hyperparams_effect(X_train, y_train, X_test, y_test, 'learning_rate', np.linspace(0.01, 0.6, 10))

# Plotting the effect of n_estimators
plot_hyperparams_effect(X_train, y_train, X_test, y_test, 'n_estimators', range(10, 110, 10))
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", -grid_search.best_score_)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Grouping and aggregating the results
grouped_results = results.groupby(['param_max_depth', 'param_n_estimators']).agg({'mean_test_score': 'mean'}).reset_index()

# Using pivot with aggregated data
pvt = grouped_results.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

# Plotting the mean test score for each combination of hyperparameters
ax = sns.heatmap(pvt, annot=True, fmt=".3f")
plt.title('Grid Search Scores')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Creating a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting each point with coordinates (max_depth, n_estimators, mean_test_score)
for index, row in results.iterrows():
    ax.scatter(row['param_max_depth'], row['param_n_estimators'], -row['mean_test_score'], c='blue', marker='o')

ax.set_xlabel('Max Depth')
ax.set_ylabel('N Estimators')
ax.set_zlabel('Mean Test Score (MSE)')
ax.set_title('3D Hyperparameter Tuning Results')

plt.show()
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=results['param_max_depth'],
    y=results['param_n_estimators'],
    z=-results['mean_test_score'],
    mode='markers',
    marker=dict(
        size=8,
        color=-results['mean_test_score'],  # set color to mean test score
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='3D Hyperparameter Tuning Results',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Mean Test Score (MSE)'
    )
)

# Show plot
fig.show()
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Prepare the data for plotting
plot_data = results.loc[:, ['param_max_depth', 'param_n_estimators', 'param_learning_rate', 'mean_test_score']]
plot_data['mean_test_score'] = -plot_data['mean_test_score']

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=plot_data['param_max_depth'],
    y=plot_data['param_n_estimators'],
    z=plot_data['param_learning_rate'],
    mode='markers',
    marker=dict(
        size=6,
        color=plot_data['mean_test_score'],  # set color to mean test score
        colorscale='Viridis',  # choose a colorscale
        colorbar_title='MSE',
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='3D Hyperparameter Tuning Results with XGBoost',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Learning Rate'
    ),
    scene_aspectmode='cube'
)

# Show plot
fig.show()
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a finer parameter grid
param_grid = {
    'max_depth': range(3, 10),
    'n_estimators': range(50, 200, 25),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
}

# Initialize the XGBRegressor with GPU acceleration
xgb_reg = XGBRegressor(tree_method='gpu_hist', gpu_id=0)

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Prepare the data for plotting
plot_data = results.loc[:, ['param_max_depth', 'param_n_estimators', 'param_learning_rate', 'mean_test_score']]
plot_data['mean_test_score'] = -plot_data['mean_test_score']

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=plot_data['param_max_depth'],
    y=plot_data['param_n_estimators'],
    z=plot_data['param_learning_rate'],
    mode='markers',
    marker=dict(
        size=4,
        color=plot_data['mean_test_score'],  # set color to mean test score
        colorscale='Viridis',  # choose a colorscale
        colorbar_title='MSE',
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='Fine-Grained 3D Hyperparameter Tuning Results with XGBoost (GPU Accelerated)',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Learning Rate'
    ),
    scene_aspectmode='cube'
)

# Show plot
fig.show()
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a finer parameter grid
param_grid = {
    'max_depth': range(3, 10),
    'n_estimators': range(50, 200, 25),
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
}

# Initialize the XGBRegressor with GPU acceleration
xgb_reg = XGBRegressor(device='gpu', gpu_id=0)

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Prepare the data for plotting
plot_data = results.loc[:, ['param_max_depth', 'param_n_estimators', 'param_learning_rate', 'mean_test_score']]
plot_data['mean_test_score'] = -plot_data['mean_test_score']

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=plot_data['param_max_depth'],
    y=plot_data['param_n_estimators'],
    z=plot_data['param_learning_rate'],
    mode='markers',
    marker=dict(
        size=4,
        color=plot_data['mean_test_score'],  # set color to mean test score
        colorscale='Viridis',  # choose a colorscale
        colorbar_title='MSE',
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='Fine-Grained 3D Hyperparameter Tuning Results with XGBoost (GPU Accelerated)',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Learning Rate'
    ),
    scene_aspectmode='cube'
)

# Show plot
fig.show()
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a dense parameter grid
param_grid = {
    'max_depth': range(3, 10),             # Exploring depths from 3 to 9
    'n_estimators': range(50, 201, 10),    # Using steps of 25 from 50 to 200
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]  # Exploring several learning rates
}

# Initialize the XGBRegressor
xgb_reg = XGBRegressor()

# Initialize the Grid Search model
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Convert the grid search results to a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Prepare the data for plotting
plot_data = results.loc[:, ['param_max_depth', 'param_n_estimators', 'param_learning_rate', 'mean_test_score']]
plot_data['mean_test_score'] = -plot_data['mean_test_score']

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=plot_data['param_max_depth'],
    y=plot_data['param_n_estimators'],
    z=plot_data['param_learning_rate'],
    mode='markers',
    marker=dict(
        size=4,
        color=plot_data['mean_test_score'],  # set color to mean test score
        colorscale='Viridis',  # choose a colorscale
        colorbar_title='MSE',
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='Dense Grid Hyperparameter Tuning Results with XGBoost',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Learning Rate'
    ),
    scene_aspectmode='cube'
)

# Show plot
fig.show()
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Assuming `results` is the DataFrame obtained from GridSearchCV
plot_data = results.loc[:, ['param_max_depth', 'param_n_estimators', 'param_learning_rate', 'mean_test_score']]
plot_data['mean_test_score'] = -plot_data['mean_test_score']

# Preparing data for interpolation
points = plot_data[['param_max_depth', 'param_n_estimators', 'param_learning_rate']].values
values = plot_data['mean_test_score'].values

# Creating a meshgrid for interpolation
grid_x, grid_y, grid_z = np.mgrid[3:10:100j, 50:200:100j, 0.01:0.2:100j]

# Interpolating data
grid_data = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')

# Create a 3D scatter plot
fig = go.Figure(data=[go.Volume(
    x=grid_x.flatten(),
    y=grid_y.flatten(),
    z=grid_z.flatten(),
    value=grid_data.flatten(),
    isomin=grid_data.min(),
    isomax=grid_data.max(),
    opacity=0.3,  # Lower opacity to see through
    surface_count=20,  # The number of isosurface levels
    colorscale='Viridis'
)])

# Set plot layout
fig.update_layout(
    title='Continuous 3D Hyperparameter Tuning Results with XGBoost',
    scene=dict(
        xaxis_title='Max Depth',
        yaxis_title='N Estimators',
        zaxis_title='Learning Rate'
    )
)

# Show plot
fig.show()
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Varying alpha for L1 regularization
alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for alpha in alpha_values:
    model = xgb.XGBRegressor(objective='reg:squarederror', alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Alpha: {alpha}, MSE: {mse}")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Varying alpha for L1 regularization and capturing MSE
alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mse_scores_alpha = []
for alpha in alpha_values:
    model = xgb.XGBRegressor(objective='reg:squarederror', alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores_alpha.append(mse)

# Plotting the MSE for different values of alpha
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, mse_scores_alpha, marker='o', linestyle='-', color='b')
plt.title('Effect of L1 Regularization (Alpha) on MSE')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
# Varying lambda for L2 regularization
lambda_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for lambda_val in lambda_values:
    model = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=lambda_val)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Lambda: {lambda_val}, MSE: {mse}")
# Varying lambda for L2 regularization and capturing MSE
lambda_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mse_scores_lambda = []
for lambda_val in lambda_values:
    model = xgb.XGBRegressor(objective='reg:squarederror', reg_lambda=lambda_val)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores_lambda.append(mse)

# Plotting the MSE for different values of lambda
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, mse_scores_lambda, marker='o', linestyle='-', color='r')
plt.title('Effect of L2 Regularization (Lambda) on MSE')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Example using the Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Cost complexity pruning
path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train multiple trees with different alpha values
trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    trees.append(clf)

# Evaluate each tree and select the one with the best trade-off
best_tree = None
best_score = 0
for tree in trees:
    y_pred = tree.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_tree = tree

# Plotting accuracy vs alpha for visualization
test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in trees]
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, test_scores, marker='o', drawstyle="steps-post")
plt.xlabel("alpha")
plt.ylabel("accuracy")
plt.title("Accuracy vs alpha for training set")
plt.show()

print("Best tree ccp_alpha:", best_tree.ccp_alpha)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Cost complexity pruning
path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train multiple trees with different alpha values
trees = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    trees.append(clf)

# Visualize the pruned trees
fig, axes = plt.subplots(nrows=1, ncols=len(trees), figsize=(len(trees) * 2, 4), dpi=100)

for i in range(len(trees)):
    plot_tree(trees[i], ax=axes[i], filled=True)
    axes[i].set_title(f'Tree with alpha={ccp_alphas[i]:.4f}')

plt.tight_layout()
plt.show()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Load iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Cost complexity pruning
path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Select a subset of ccp_alphas for visualization (first 4 unique values)
unique_ccp_alphas = np.unique(ccp_alphas)[:4]

# Train multiple trees with different alpha values
trees = []
for ccp_alpha in unique_ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    trees.append(clf)

# Visualize the pruned trees (first 4)
fig, axes = plt.subplots(nrows=1, ncols=len(trees), figsize=(len(trees) * 4, 4), dpi=100)

for i in range(len(trees)):
    plot_tree(trees[i], ax=axes[i], filled=True)
    axes[i].set_title(f'Tree with alpha={unique_ccp_alphas[i]:.4f}')

plt.tight_layout()
plt.show()
from sklearn.tree import DecisionTreeRegressor

# Train a decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Cost complexity pruning
path = tree_reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train multiple regression trees with different alpha values
reg_trees = []
for ccp_alpha in ccp_alphas:
    reg = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    reg.fit(X_train, y_train)
    reg_trees.append(reg)

# Select the tree with the best balance between complexity and prediction accuracy
# [Above Code to select the best tree based on validation set or other criteria
from sklearn.metrics import mean_squared_error

# Split the original training data into new training and validation sets
X_train_new, X_val, y_train_new, y_val = train_test_split(
X_train, y_train, test_size=0.3, random_state=42)
# Evaluate each pruned tree on the validation set
mse_values = []
for reg_tree in reg_trees:
    y_val_pred = reg_tree.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    mse_values.append(mse)

# Select the tree with the lowest MSE value
optimal_tree_index = mse_values.index(min(mse_values))
optimal_tree = reg_trees[optimal_tree_index]

print(f"Optimal CCP alpha: {ccp_alphas[optimal_tree_index]}")
print(f"Minimum MSE on validation set: {min(mse_values)}")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Cost complexity pruning
path = tree_reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Select a subset of ccp_alphas for visualization (e.g., first 4 unique values)
selected_ccp_alphas = np.unique(ccp_alphas)[:4]

# Train multiple regression trees with different alpha values
reg_trees = []
for ccp_alpha in selected_ccp_alphas:
    reg = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    reg.fit(X_train, y_train)
    reg_trees.append(reg)

# Visualize the pruned trees
fig, axes = plt.subplots(nrows=1, ncols=len(reg_trees), figsize=(len(reg_trees) * 4, 4), dpi=100)

for i in range(len(reg_trees)):
    plot_tree(reg_trees[i], ax=axes[i], filled=True)
    axes[i].set_title(f'Tree with alpha={selected_ccp_alphas[i]:.4f}')

plt.tight_layout()
plt.show()

