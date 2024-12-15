# %%
# Loading packages
from project_d100.data import load_parquet
from project_d100.modelling import glm_pipeline, glm_tuning, lgbm_pipeline, lgbm_tuning
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
predictors = [col for col in X_train.columns if col != "instant"]
categoricals = [
    "season",
    "yr",
    "mnth",
    "hr",
    "weekday",
    "workingday",
    "weathersit",
]
numericals = [col for col in predictors if col not in categoricals]

# %%
# Defining pipelines
glm_pipeline = glm_pipeline(numericals, categoricals)
lgbm_pipeline = lgbm_pipeline(numericals, categoricals)

# %%
# Tuning the GLM model
glm_best = glm_tuning(X_train, y_train, glm_pipeline)
# Model finds unknown category during transform from weathersit. Likely weathersit=4.
# best alpha: 0.01
# best l1_ratio: 0.75 (ridge regression)

# %%
# Tuning the LGBM model
lgbm_best = lgbm_tuning(X_train, y_train, lgbm_pipeline)
# best n_leaves: 31
# best n_estimators: 200
# best min_child_weight: 1
# best learning_rate: 0.1
# %%
