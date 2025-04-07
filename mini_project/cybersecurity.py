# %%
from os.path import join as path_join

import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
from holoviews.ipython import display
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import zscore
from dataset import df

# %%
data_root = kagglehub.dataset_download("andrewkronser/cve-common-vulnerabilities-and-exposures")

# %%
## 📌 1. Loading the Dataset

# >## 💡 **Interpretation**:
# -  **mod_date: The date the entry was last modified.**
# -  **pub_date: The date the entry was published.**
# -  **cvss: Common Vulnerability Scoring System (CVSS) score, a measure of the severity of a vulnerability.**
# -  **cwe_code: Common Weakness Enumeration (CWE) code, identifying the type of weakness.**
# -  **cwe_name: The name associated with the CWE code.**
# -  **summary: A text summary of the vulnerability.**
# -  **access_authentication.**
# -  **access_complexity: how difficult it is to execute.**
# -  **access_vector: how the attack is performed, aka via network or locally.**

# df = pd.read_csv(path_join(data_root, 'cve.csv'), header=0, index_col=0)
df.mod_date = pd.to_datetime(df.mod_date)
df.pub_date = pd.to_datetime(df.pub_date)

# Dataset summary
print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

non_par_com = {"NONE": 0, "COMPLETE": 2, "PARTIAL": 1}
low_med_hih = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
non_sin_mul = {"NONE": 0, "SINGLE": 1, "MULTIPLE": 2}
loc_adj_net = {"LOCAL": 0, "ADJACENT_NETWORK": 1, "NETWORK": 2}

ordinal_remapping = {
    "access_authentication": non_sin_mul,
    "access_complexity": low_med_hih,
    "access_vector": loc_adj_net,
    "impact_availability": non_par_com,
    "impact_confidentiality": non_par_com,
    "impact_integrity": non_par_com,
}

for ordinal_column in ordinal_remapping:
    df[ordinal_column] = df[ordinal_column].apply(
        lambda v: ordinal_remapping[ordinal_column].get(v, v)
    )

# %%
## 🧼 2. Handling Missing Data

missing_counts = df.isnull().sum()
print("Missing Data Count:\n", missing_counts)



# %%
## 📐 3. Probability Distribution & Descriptive Stats


# The McCumber Cube is a model framework created by John McCumber in 1991 to
# help organizations establish and evaluate information security initiatives
# by considering all of the related factors that impact them.
# This security model has three dimensions:
# The foundational principles for protecting information systems.
# 1.availability, 2.integrity, 3.Confidentiality ✅
# The protection of information in each of its possible states.
# The security measures used to protect data.


### 3.1 Minimum and Maximum Values

# Min and Max of DataFrame columns
print("Minimum values in each column:")
print(df.min(numeric_only=True))

print("\nMaximum values in each column:")
print(df.max(numeric_only=True))


# %%
### 3.2 Geometric Mean
# Calculate the geometric mean of cvss
geometric_mean_cvss = stats.gmean(df["cvss"].dropna())
print(f"\nGeometric Mean of cvss: {geometric_mean_cvss:.2f}")

# Calculate the geometric mean of cwe_code
geometric_mean_cwe_code = stats.gmean(df["cwe_code"].dropna())
print(f"\nGeometric Mean of cwe_code: {geometric_mean_cwe_code:.2f}")


# %%
### 3.3 Median and Mode

# Calculate median impact_availability
median_impact_availability = np.median(df["impact_availability"] / ((df["impact_availability"] / 100) ** 2))
print(f"Median availability: {median_impact_availability:.2f}")

# Calculate the mode of impact_integrity
mode_impact_integrity = stats.mode(df["impact_integrity"], keepdims=True)
print(f"Most Common integrity: {mode_impact_integrity.mode[0]}, Count: {mode_impact_integrity.count[0]}")



# %%
### 3.4 Variance and Standard Deviation


# %%
### 3.5 Quantiles and Interquartile Range (IQR)

# Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR for impact_confidentiality
q1_confidentiality = df["impact_confidentiality"].quantile(0.25)
q3_confidentiality = df["impact_confidentiality"].quantile(0.75)
iqr_confidentiality = q3_confidentiality - q1_confidentiality

print(f"Q1 (25th percentile of confidentiality): {q1_confidentiality:.2f}")
print(f"Q3 (75th percentile of confidentiality): {q3_confidentiality:.2f}")
print(f"Interquartile Range (IQR) of confidentiality: {iqr_confidentiality:.2f}")

# Visualize IQR using a boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["impact_confidentiality"])
plt.title("Boxplot of Confidentiality Distribution")
plt.show()


# %%
# ## 📊 4.Visualising Relationships

### 📊 4.1 Distribution Shapes

column_name = 'impact_confidentiality'

# Drop NaN values just in case
clean_impact_confidentiality = df[column_name].dropna()

# Mean, Variance, and Standard Deviation
mean_val = np.mean(clean_impact_confidentiality)
var_val = np.var(clean_impact_confidentiality, ddof=1)     # sample variance
std_val = np.std(clean_impact_confidentiality, ddof=1)     # sample standard deviation

print(f"Mean (Expected Value): {mean_val:.4f}")
print(f"Variance: {var_val:.4f}")
print(f"Standard Deviation: {std_val:.4f}")

# Visualize distribution
plt.hist(clean_impact_confidentiality, bins=30)
plt.title(f"Histogram of {column_name}")
plt.xlabel(column_name)
plt.ylabel("Frequency")
plt.show()

# Summary:
# The histogram shows the distribution of the impact_confidentiality values in our dataset.
#
# We also calculated key descriptive statistics:
#
# - The mean value of approximately 0.8659 indicates the average confidentiality values in the dataset.
# - The variance of 0.5085 tells us how spread out the values are — specifically, it’s the average of the squared differences from the mean.
# - The standard deviation of 0.7131 shows that most of the confidentiality values are within ±0.7131 units of the mean —
# a more interpretable measure than variance since it shares the same units as the data.
# - The histogram is asymmetric and slightly skewed, suggesting that the distribution is not perfectly normal.
# - There's a wide spread in the values, ranging from slightly below 15000 to above 40000.


# %%
## 📊 4.2 Poisson distribution

# Estimate the mean of access_complexity
lambda_value = df["access_complexity"].mean()

# Generate Poisson distribution
poisson_data = np.random.poisson(lam=lambda_value, size=1000)

# Plot distribution
sns.histplot(poisson_data, kde=True, bins=30, color="blue")
plt.title("Poisson Distribution of access_complexity")
plt.xlabel("access_complexity")
plt.ylabel("Frequency")
plt.show()


# %%
### 📦 4.3 Detect, Report, and Visualize Outliers Using Z-Score

# Detect, Report, and Visualize Outliers Using Z-Score

def visualize_outliers(df, threshold=3):
    df_numeric = df.select_dtypes(include=['number'])

    # Calculate Z-scores
    z_scores = df_numeric.apply(zscore, nan_policy='omit')

    # Count how many values are considered outliers
    outlier_counts = (z_scores.abs() > threshold).sum()
    print("Number of outliers detected per column:\n", outlier_counts)

    # Summary statistics
    print("\n--- Summary Statistics ---")
    display(df_numeric.describe().T)

    # Boxplot visualization
    print("\n Boxplots to Inspect Outliers:")
    df_numeric.plot(kind='box', subplots=True, layout=(1, len(df_numeric.columns)), figsize=(16, 4), patch_artist=True)
    plt.tight_layout()
    plt.show()

# Apply outlier visualization
visualize_outliers(df)

# %%
## 🔗 5. Analysing Correlation Between Variables

# Covariance Analysis

# Calculate a rolling mean to smooth out short-term fluctuations
df['rolling_mean'] = df['access_authentication'].rolling(window=30).mean()

# Remove rows with NaN values introduced by the rolling window
df_clean = df.dropna()

# Calculate the covariance matrix between the original values and their rolling mean
cov_matrix = df_clean[['access_authentication', 'rolling_mean']].cov()

# Display the covariance matrix
print("\nCovariance Matrix:")
print(cov_matrix)

# Visualize the covariance matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", center=0)

# Add plot title
plt.title("Covariance Matrix Heatmap")

# Show the plot
plt.show()


# %%
# Calculate correlation matrices using different methods:

pearson_corr = cov_matrix.corr(method='pearson') # - Pearson: linear correlation (assumes normality)
spearman_corr = cov_matrix.corr(method='spearman') # - Spearman: rank-based correlation (monotonic relationships)
kendall_corr = cov_matrix.corr(method='kendall') # - Kendall: rank correlation (more robust with small samples or ties)

# Display the correlation matrices
print("\nPearson Correlation:")
print(pearson_corr)

print("\nSpearman Correlation:")
print(spearman_corr)

print("\nKendall Correlation:")
print(kendall_corr)

plt.figure(figsize=(6, 4))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", center=0)

# Add title
plt.title("Pearson Correlation Heatmap")

# Show plot
plt.show()
