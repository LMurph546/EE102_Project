import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import mean_squared_error, r2_score

#STEP 1: is to decide on project topic and variables, which we chose to do
#life expectancy and its correlation with GDP, daily income, CO2 emissions,
#children per woman, and child mortality. 
# We will use the 2013 data for all of these variables


#STEP 2: Load and Prepare Datasets




df_life = pd.read_csv('life_expectancy.csv')
df_gdp = pd.read_csv('gdp_percap.csv')
df_income = pd.read_csv('daily_income.csv')
df_co2 = pd.read_csv('co2_percap.csv')
df_children = pd.read_csv('children_per_woman.csv')
df_child_mortality = pd.read_csv('child_morality_ages_0to5.csv')

df_life = df_life.rename(columns={'Value(2013)': 'Life_Expectancy'})
df_gdp = df_gdp.rename(columns={'Value(2013)': 'GDP'})
df_income = df_income.rename(columns={'Value(2013)': 'Daily_Income'})
df_co2 = df_co2.rename(columns={'Value(2013)': 'CO2_Emissions'})
df_children = df_children.rename(columns={'Value(2013)': 'Children_per_Woman'})
df_child_mortality = df_child_mortality.rename(columns={'Value(2013)': 'Child_Mortality'})

#include year to all of the variables
for df_ in [df_life, df_gdp, df_income, df_co2, df_children, df_child_mortality]:
    df_['Year'] = 2013




#Merge the datasets on Country and Year
df = df_life.merge(df_gdp, on=['Country', 'Year'])
df = df.merge(df_income, on=['Country', 'Year'])
df = df.merge(df_co2, on=['Country', 'Year'])
df = df.merge(df_children, on=['Country', 'Year'])
df = df.merge(df_child_mortality, on=['Country', 'Year'])





#STEP 3: Data Cleaning and Summary Statistics



#  Function to Clean "k" formatted strings to numeric values
# This function converts strings with 'k' to numeric values (e.g., '1.5k' to 1500)
def convert_k_string(value):
    try:
        val = str(value).lower().strip()
        if 'k' in val:
            return float(val.replace('k', '')) * 1000
        return float(val)
    except:
        return np.nan

variables = ['Life_Expectancy', 'GDP', 'Daily_Income', 'CO2_Emissions', 'Children_per_Woman', 'Child_Mortality']
for var in variables:
    df[var] = df[var].apply(convert_k_string)

# === Function: Get Summary Statistics ===
def get_summary(data, vars_list):
    summary = {}
    for var in vars_list:
        summary[var] = {
            'Mean': data[var].mean(),
            'Median': data[var].median(),
            'Mode': data[var].mode().iloc[0] if not data[var].mode().empty else None,
            'Min': data[var].min(),
            'Q1': data[var].quantile(0.25),
            'Q3': data[var].quantile(0.75),
            'Max': data[var].max(),
            'Variance': data[var].var(),
            'Std_Dev': data[var].std()
        }
    return pd.DataFrame(summary).T.round(2)

# === Summary BEFORE trimming ===
summary_untrimmed = get_summary(df, variables)
print("\nSummary Statistics (Untrimmed Data):")
print(summary_untrimmed)

# === Trim outliers per variable ===
df_trimmed = df.copy()
for var in variables:
    z = zscore(df_trimmed[var].dropna())
    mask = np.abs(z) < 3
    trimmed = df_trimmed[var].dropna()[~mask].count()
    print(f"Trimmed {trimmed} outliers from {var}")
    df_trimmed.loc[df_trimmed[var].dropna().index[~mask], var] = np.nan

df_trimmed = df_trimmed.dropna()
print(f"\nRows remaining after trimming: {df_trimmed.shape[0]}\n")

# === Summary AFTER trimming ===
summary_trimmed = get_summary(df_trimmed, variables)
print("Summary Statistics (Trimmed Data):")
print(summary_trimmed)

# Save both summaries to CSV files
summary_untrimmed.to_csv("summary_untrimmed.csv")
summary_trimmed.to_csv("summary_trimmed.csv")



# Create a pair plot
pair_plot = sns.pairplot(df[variables], corner=True, diag_kind='kde')

# Increase the figure size
pair_plot.figure.set_size_inches(12, 10)

# Add padding at the bottom to avoid cutting off axis labels
pair_plot.figure.subplots_adjust(bottom=0.1, top=0.95)

# Optional: add a title
pair_plot.figure.suptitle("Scatter Plot Matrix (Untrimmed Data)", fontsize=16)

plt.show()



# Compute correlation matrix
correlation_matrix = df[variables].corr()

# Show correlation of all variables with Life Expectancy
print("\n Correlation with Life Expectancy:")
print(correlation_matrix['Life_Expectancy'].sort_values(ascending=False))


#for the remainder of the project, we use Child_Morality, Children_per_Women, and GDP as they
#are the most correlated with Life Expectancy
#Correlation with Life Expectancy:
#GDP                   0.666888
#Daily_Income          0.664457
#CO2_Emissions         0.544967
#Children_per_Woman   -0.803966
#Child_Mortality      -0.854217

#=== Correlation Between Input Variables ===

# Choose the 3 input variables you’re working with
input_vars = ['Child_Mortality', 'Children_per_Woman', 'GDP']

# Correlation matrix
corr_inputs = df[input_vars].corr()

# Print correlation matrix
print("\n Correlation Between Selected Input Variables:")
print(corr_inputs)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_inputs, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Between Input Variables")
plt.tight_layout()
plt.show()

pair_plot.figure.savefig("pairplot_variables.png", bbox_inches='tight')
plt.savefig("heatmap_inputs.png", bbox_inches='tight')




#STEP 4: Future Nominalization
# Create a copy of just the input variables
df_normalized = df[input_vars].copy()

# Apply min-max normalization safely
for var in input_vars:
    min_val = df_normalized[var].min()
    max_val = df_normalized[var].max()
    range_val = max_val - min_val

    if range_val != 0:
        df_normalized[var] = (df_normalized[var] - min_val) / range_val
    else:
        df_normalized[var] = 0.0  # All values are the same — set to 0

print("\nNormalized Input Variables (0 to 1 range):")
print(df_normalized.head())



#STEP 5: Data Splitting
# Split the data into training and testing sets

#define inputs X and output Y
X = df_normalized[input_vars]
Y = df['Life_Expectancy'] #not normalzied as it is the output variable

# Split into training and testing sets (90/10 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
#42 is the random state for reproducibility, hence why we use it

# Print the sizes of the training and testing sets
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")





#Step 6: Derive the linear models to predict the output variable

def derive_linear_model(X, Y):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    coeffs, residuals, rank, s = np.linalg.lstsq(X_aug, Y, rcond=None)
    return coeffs

# === Store all models and results ===


#the more variables we add, the more accurate the model will be. We start with 1-variable models
#and work our way up to 3-variable models, which is the most accurate model. 
models = []

# === 1-variable models ===
print("\n 1-Variable Models")
for var in input_vars:
    X1 = X_train[[var]].to_numpy()
    Y1 = Y_train.to_numpy()
    coeffs = derive_linear_model(X1, Y1)
    print(f"{var} → Y = {coeffs[0]:.4f} + {coeffs[1]:.4f}·{var}")
    models.append(([var], coeffs))

# === 2-variable models ===
print("\n 2-Variable Models")
for var_combo in itertools.combinations(input_vars, 2):
    X2 = X_train[list(var_combo)].to_numpy()
    Y2 = Y_train.to_numpy()
    coeffs = derive_linear_model(X2, Y2)
    print(f"{var_combo} → Y = {coeffs[0]:.4f} + {coeffs[1]:.4f}·{var_combo[0]} + {coeffs[2]:.4f}·{var_combo[1]}")
    models.append((list(var_combo), coeffs))

# === 3-variable model ===
print("\n 3-Variable Model")
X3 = X_train[input_vars].to_numpy()
Y3 = Y_train.to_numpy()
coeffs = derive_linear_model(X3, Y3)
print(f"{input_vars} → Y = {coeffs[0]:.4f} + {coeffs[1]:.4f}·{input_vars[0]} + {coeffs[2]:.4f}·{input_vars[1]} + {coeffs[3]:.4f}·{input_vars[2]}")
models.append((input_vars, coeffs))











#STEP 7: Evaluate the models
print("\n Model Evaluation on Test Data (Using MSE and R²)")

# Loop through models
for input_features, coeffs in models:
    X_test_model = X_test[input_features].to_numpy()
    X_test_aug = np.hstack([np.ones((X_test_model.shape[0], 1)), X_test_model])
    
    Y_pred = X_test_aug @ coeffs
    Y_actual = Y_test.to_numpy()

    mse = mean_squared_error(Y_actual, Y_pred)
    r2 = r2_score(Y_actual, Y_pred)

    print(f"\nModel {input_features}:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")


    # Print predicted vs actual values side-by-side
print("\n Predicted vs Actual (first 10 rows):")
for pred, actual in zip(Y_pred[:10], Y_actual[:10]):
    print(f"  Predicted: {pred:.2f} \t Actual: {actual:.2f}")

# Adjust the layout depending on how many models you have
n_models = len(models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

# Create the figure and axes grid
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axs = axs.flatten()  # Flatten in case we have fewer than full rows

for i, (input_features, coeffs) in enumerate(models):
    X_test_model = X_test[input_features].to_numpy()
    X_test_aug = np.hstack([np.ones((X_test_model.shape[0], 1)), X_test_model])
    Y_pred = X_test_aug @ coeffs
    Y_actual = Y_test.to_numpy()

    ax = axs[i]
    ax.scatter(Y_actual, Y_pred, edgecolor='k', alpha=0.7)
    ax.plot([Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], 'r--')
    ax.set_xlabel("Actual", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.set_title(f"Model: {', '.join(input_features)}", fontsize=10)
    ax.grid(True)

# Turn off unused subplots if models < n_rows * n_cols
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

# Adjust layout spacing
fig.tight_layout(pad=2.0)
fig.suptitle("Predicted vs Actual Life Expectancy (All Linear Models)", fontsize=16, y=0.95)
plt.subplots_adjust(top=0.9)  # Add space for the suptitle

plt.show()














#STEP 8: Heuristic Model
# The heuristic model is a custom function that predicts life expectancy based on GDP, fertility rate, and child mortality

# === Step 1: Define the heuristic function ===
def heuristic_model(row):
    gdp = row['GDP']
    fertility = row['Children_per_Woman']
    mortality = row['Child_Mortality']
    
    # Prevent math errors
    gdp = max(gdp, 1)
    mortality = max(mortality, 1)
    
    # Custom heuristic function (feel free to tweak coefficients)
    return 70 + 12 * np.log(gdp) - 4 * fertility - 0.15 * mortality

# === Step 2: Apply it to the test set ===
Y_pred_heuristic = X_test.apply(heuristic_model, axis=1)
Y_actual = Y_test

# === Step 3: Compute MSE ===
mse_heuristic = mean_squared_error(Y_actual, Y_pred_heuristic)
print(f"\n Heuristic Model MSE: {mse_heuristic:.4f}")

# === Step 4: Print first few predictions ===
print("\n Heuristic Model — Predicted vs Actual (first 10):")
for pred, actual in zip(Y_pred_heuristic[:10], Y_actual[:10]):
    print(f"  Predicted: {pred:.2f} \t Actual: {actual:.2f}")

# === Step 5: Plot predicted vs actual ===
plt.figure(figsize=(6, 5))
plt.scatter(Y_actual, Y_pred_heuristic, edgecolor='k', alpha=0.7)
plt.plot([Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], 'r--')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted (Heuristic)")
plt.title("Heuristic Model: Predicted vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()