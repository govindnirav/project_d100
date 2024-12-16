# %%
# Loading packages
from project_d100.data import load_parquet
from project_d100.modelling import (
    glm_pipeline,
    glm_tuning,
    lgbm_pipeline,
    lgbm_tuning,
    save_model,
    save_split,
)
from project_d100.preprocessing import sample_split

# %%
# Loading the cleaned data
df_all = load_parquet("hours_cleaned.parquet")

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = sample_split(
    df_all, target_var="cnt", train_size=0.8, stratify="cnt", n_bins=50
)
# Saving this train test split for later use
save_split(X_train, X_test, y_train, y_test)
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
glm_best_pipeline = glm_tuning(X_train, y_train, glm_pipeline)

# %%
# Tuning the LGBM model
lgbm_best_pipeline = lgbm_tuning(X_train, y_train, lgbm_pipeline)

# %%
# Save the best pipelines for each model
save_model(glm_best_pipeline, "glm_best_pipeline")
save_model(lgbm_best_pipeline, "lgbm_best_pipeline")
