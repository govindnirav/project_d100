# %%
# Loading packages
from project_d100.data import load_parquet
from project_d100.preprocessing import sample_split

# %%
# Loading the cleaned data
df_all = load_parquet("hours_cleaned.parquet")

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = sample_split(
    df_all, target_var="cnt", train_size=0.8, stratify="cnt", n_bins=50
)
# Tried to stratify by cnt but too few observations in some classes,
# so stratified by bin instead

# %%
# Classifying predictors
Predictors = X_train.columns.difference(["instance"]).tolist()
Categoricals = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
Numericals = [col for col in Predictors if col not in Categoricals]

# %%
