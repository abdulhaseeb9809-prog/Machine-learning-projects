import streamlit as st
import pandas as pd
import pickle # Pickle used for saving the model

from automl_code import run_automl_unified, predict_new # Calling the auto ml for training and selecting best model and predict_new for predicting new prediction

st.title("Unified AutoML App") # Inserting a title

uploaded_file = st.file_uploader("Upload CSV Dataset")  # this is used to create an Upload button in UI

if uploaded_file is not None: # Only run next code AFTER user uploads file

    df = pd.read_csv(uploaded_file) # Reading the Uploaded file

    st.subheader("Dataset Preview") # Gives a subheading Dataset Preview
    st.dataframe(df.head()) # Shows the first 5 rows of the data set

    target = st.selectbox("Select Target Column", df.columns) # This is an option for users to select target variable themselves

    if st.button("Run AutoML"): # Creating a button for running auto ml so that the model can find the best model

        model, ranking, problem, label_encoder, feature_importance = run_automl_unified(df, target) #Runs the auto ml main pipeline and gets the model ranking problem, LE, and featur imp

        st.subheader("Model Ranking") # Shows model ranking
        st.dataframe(ranking)

        st.success(f"Detected Problem Type: {problem}") # Shows detected problem type

        X = df.drop(columns=[target]) # Preparing input for predictions

        results = predict_new(model, label_encoder, problem, X.head()) # USing the predict_new pipeline predicting the first 5 rows of the dat

        st.subheader("Predictions (First 5 Rows)") # Shows the result of predictions
        st.write(results["Predictions"])

        if "Probabilities" in results:
            st.subheader("Probabilities")
            st.write(results["Probabilities"]) # If there is probabilities then it shows that too

        # Feature Importance
        st.subheader("Feature Importance") 

        if feature_importance is not None: # Only runs if  the feature imp is not none
            st.bar_chart(feature_importance.set_index("Feature")) #Creating a bar chart of feature importance
        else:
            st.write("Feature importance not available for this model") # IF no Feature Importance it returns this

        # Download Model
        st.subheader("💾 Download Trained Model") 

        model_bytes = pickle.dumps(model)  # Converts the model in to binary format for download

        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="automl_model.pkl"
        ) # Creates a button for downloading the pickel model