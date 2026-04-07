import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBClassifier, XGBRegressor


def detect_problem_type(y): # In here we are trying to detect the problem
    if y.nunique() <= 20: #If number of unique character is less than 20 means classification else regression
        return "classification"
    return "regression" # Creating a udf function to later use it in final pipeline


def tune_model(pipeline, param_grid, X_train, y_train): # Hyperparameter tuning using RandomizedSearchCV

    search = RandomizedSearchCV(
        pipeline, # Preprocessing + Model
        param_grid, # Parameters to tune
        n_iter=5, # Try 5 Combinations
        cv=3, #Cross validation 3 times
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train) #Tries different parameter combinations  AND Finds best one

    return search.best_estimator_ # Returns best tuned model


def run_automl_unified(df, target_column):

    print("🚀 Starting AutoML...")

    X = df.drop(columns=[target_column]) # Spliting dependent and independent variable
    y = df[target_column]

    problem_type = detect_problem_type(y) # The previously defined function is used here returns the problem type

    label_encoder = None # To later use it for classification

    # Encode classification labels
    if problem_type == "classification" and y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y) #Transforming the y variable

    print("Detected Problem Type:", problem_type) # Prints the detected problem

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist() #Finding Numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist() #Finding categorical Features
 
    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    # Preprocessing
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # Using simpleimputer to fill missing value with median
        ("scaler", StandardScaler()) #Scaling the numerical variables
    ]) 

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), # Using simpleimputer to fill missing value with mode
        ("encoder", OneHotEncoder(handle_unknown="ignore"))#Feature encoding the categorical variables
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]) # Column transformer applies different pipelines to different columns meaning rather than doing it for first category and then numerical we combine

    # Train-test split
    if problem_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        ) # Used Stratify since it is classification
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        ) # No stratify here

    print("Train-Test split done.")

    # Models + tuning
    if problem_type == "classification": #For classification

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"), # Using class weight for class imbalance
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
            "XGBoost": XGBClassifier(eval_metric="logloss", verbosity=0)
        }

        param_grids = {
            "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]}, # model__ means that we are accessising model inside the pipeline, Since 
            "Random Forest": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10]
            },
            "XGBoost": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 6]
            }
        }

        scoring = "f1"

    else:

        models = {
            "Random Forest": RandomForestRegressor(), # For regression
            "XGBoost": XGBRegressor(verbosity=0)
        }

        param_grids = {
            "Random Forest": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10]
            },
            "XGBoost": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 6]
            }
        }

        scoring = "r2"

    ranking = []
    best_score = -np.inf #To track the BEST model score we use - infinity
    best_pipeline = None

    # Training loop
    for name, model in models.items(): # for name and model in models

        pipeline = Pipeline([
            ("preprocessor", preprocessor), # Scaling and encoding and filling missing values
            ("model", model) #Models
        ]) #Combining both

        pipeline = tune_model(pipeline, param_grids[name], X_train, y_train) #Finding the best parameters using the function we created before

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring=scoring
        ) #This is where training happens

        mean_score = scores.mean() #Taking the mean of the CV Scores

        ranking.append({
            "Model": name,
            "CV Score": mean_score
        }) #Appending to the ranking 

        if mean_score > best_score: # Is current model better than the best one so far?
            best_score = mean_score # Store new highest score
            best_pipeline = pipeline # Save this model as the BEST model

    ranking_df = pd.DataFrame(ranking).sort_values(by="CV Score", ascending=False) # Ranking the best model on cv score

    # Train best model
    best_pipeline.fit(X_train, y_train) # THe best model is then trained

    y_pred = best_pipeline.predict(X_test) # Best model is then predicted

    print("\nPerformance:")

    if problem_type == "classification": # If it is classification then check accuracy and f1 score
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    else:
        print("R2:", r2_score(y_test, y_pred))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) # Regression then R2, MAE, RMSE

    # Feature Importance
    feature_importance_df = None
    model_obj = best_pipeline.named_steps["model"] # Named steps will go inside the pipe line and access the model since pipeline.feature imp wont work

    if hasattr(model_obj, "feature_importances_"): # Check if that model has the attribute 
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out() #Getting the feature name
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model_obj.feature_importances_
        }).sort_values(by="Importance", ascending=False)

    return best_pipeline, ranking_df, problem_type, label_encoder, feature_importance_df # Returns the Complete trained model and table of ranked model and "classification" OR "regression"
# Feature importance

def predict_new(model, encoder, problem_type, df): # This function is used to predict the new data used the best trained model and using this the model gives the first 5 prediction

    preds = model.predict(df) # Predicts the new data using the final pipeline which has preprocessing and best model

    results = {}

    if problem_type == "classification":

        if encoder is not None:
            preds = encoder.inverse_transform(preds)

        results["Predictions"] = preds

        if hasattr(model.named_steps["model"], "predict_proba"):
            results["Probabilities"] = model.predict_proba(df)

    else:
        results["Predictions"] = preds

    return results # Returns the result... In our model we have used the same data set and predicted 5 rows but going forward we can use it for new input and predictions