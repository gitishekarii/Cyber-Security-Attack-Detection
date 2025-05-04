import os
import time

import streamlit as st
import pandas as pd
import pickle
import hdbscan
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from xgboost import XGBClassifier


# OPTIMIZATION to make the app faster to load data

# Create a directory to store precomputed cluster results
PRECOMPUTED_DIR = "WebApp/precomputed_clusters"
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

# Function to generate a unique hash for a dataset
def get_data_hash(df):
    """Generate a hash for the dataset to check if it has been processed before."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# Function to load precomputed clusters
def load_precomputed_clusters(data_hash):
    """Check if precomputed clusters exist for the dataset."""
    file_path = os.path.join(PRECOMPUTED_DIR, f"{data_hash}.pkl")
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

# Function to save computed clusters
def save_precomputed_clusters(df, data_hash):
    """Save cluster assignments to avoid recomputation."""
    file_path = os.path.join(PRECOMPUTED_DIR, f"{data_hash}.pkl")
    df.to_pickle(file_path)


# PRESENTATION PART

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Overall background styling */
        .main {
            background-color: #0B0C10;
            color: #C5C6C7;
            font-family: 'Arial', sans-serif;
        }

        /* Title box */
        .title-box {
            background: linear-gradient(135deg, #1F2833, #66FCF1);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 6px 20px rgba(0, 255, 255, 0.2);
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: white;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 18px;
            font-weight: 500;
            color: #45A29E;
            margin-top: 10px;
        }

        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #45A29E !important;
            background-color: rgba(69, 162, 158, 0.1);
            color: white !important;
        }

        /* Custom button */
        .stButton>button {
            background: linear-gradient(135deg, #45A29E, #66FCF1);
            color: #0B0C10;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background: #66FCF1;
            transform: scale(1.05);
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# Title with styling
st.markdown('<div class="title-box">🔐 Cybersecurity Attack Type Prediction ML Model 🔍</div>',
            unsafe_allow_html=True)

# Subtitle
st.markdown(
    '<p class="subtitle">Upload your dataset to identify the best ML model for predicting potential cyber threats.</p>',
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True) # Space for separation

# File uploader with a more engaging instruction
uploaded_file = st.file_uploader(
    "📂 **Upload your pre-processed CSV file:** (encoded & normalized)",
    type="csv"
)


# Checking if a file has been uploaded
if uploaded_file is not None:

    # Reading the CSV file into a DataFrame
    df_cyber_processed = pd.read_csv(uploaded_file)

    # If only one column, try reading with semicolon delimiter
    if df_cyber_processed.shape[1] == 1:
        uploaded_file.seek(0)  # Reset file pointer
        df_cyber_processed = pd.read_csv(uploaded_file, delimiter=";")

    st.write("Processed Dataset preview:")
    st.dataframe(df_cyber_processed.head(25))


    # TRAINING PART

    # we drop 'User Information' column from the dataset
    df_cyber_processed = df_cyber_processed.drop(columns=['User Information'], errors='ignore')

    # Separate the features and the target variable
    X = df_cyber_processed.drop('Attack Type', axis=1)
    y = df_cyber_processed['Attack Type']

    # For the 2 classic models
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42)

    # For the 2 cluster models
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.25, random_state=42,stratify=y_train_val)


    # MODELING PART

    # creation of tabs for each model
    tab1, tab2, tab3, tab4 = st.tabs([
        "Logistic Regression",
        "K-Nearest Neighbors",
        "HDBSCAN",
        "HDBSCAN & XGBoost"
    ])

    with tab1:
        st.title("Logistic Regression")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/logistic_regression_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test_1)

        # Define mapping for attack types
        attack_mapping = {0: "DDoS", 1: "Intrusion", 2: "Malware"}

        # Apply mapping to predictions and actual values
        df_predictions = pd.DataFrame({
            'Actual Attack Type': [attack_mapping[label] for label in y_test_1],
            'Predicted Attack Type': [attack_mapping[label] for label in y_pred]
        })

        # Display the predictions in Streamlit
        st.subheader("Predictions")
        st.dataframe(df_predictions.head(25))

        # Calculate accuracy
        acc = accuracy_score(y_test_1, y_pred)

        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test_1, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")


    with tab2:
        st.title("K-Nearest Neighbors")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/k-nearest_neighbors_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test_1)

        # Define mapping for attack types
        attack_mapping = {0: "DDoS", 1: "Intrusion", 2: "Malware"}

        # Apply mapping to predictions and actual values
        df_predictions = pd.DataFrame({
            'Actual Attack Type': [attack_mapping[label] for label in y_test_1],
            'Predicted Attack Type': [attack_mapping[label] for label in y_pred]
        })

        # Display the predictions in Streamlit
        st.subheader("Predictions")
        st.dataframe(df_predictions.head(25))

        # Calculate accuracy
        acc = accuracy_score(y_test_1, y_pred)

        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test_1, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")


    with tab3:
        st.title("HDBSCAN")

        # Define target and features
        target = 'Attack Type'
        X = df_cyber_processed.drop(columns=[target])  # Features
        y = df_cyber_processed[target]  # Target

        # Generate dataset hash
        data_hash = get_data_hash(X)

        # Check if clusters were already computed for this dataset
        precomputed_data = load_precomputed_clusters(data_hash)

        if precomputed_data is not None:
            df_cyber_processed = precomputed_data
            is_precomputed = True
        else:
            is_precomputed = False

            # Loading trained HDBSCAN model
            with open("WebApp/trained_models/best_hdbscan_model.pkl", "rb") as model_file:
                hdbscan_model = pickle.load(model_file)

            # Identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

            # Encoding categorical features
            label_encoders = {col: LabelEncoder() for col in categorical_cols}
            for col in categorical_cols:
                X[col] = label_encoders[col].fit_transform(X[col])

            # Normalize numerical features
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

            # Convert to numpy array for HDBSCAN
            X_array = X.to_numpy()

            # Predict clusters using the trained model
            clusters = hdbscan_model.fit_predict(X_array)
            df_cyber_processed['Cluster'] = clusters

            # Map clusters to the most common Attack_Type
            cluster_mapping = df_cyber_processed.groupby('Cluster')[target].agg(lambda x: x.mode()[0]).reset_index()
            cluster_mapping.columns = ['Cluster', 'Predicted_Attack_Type']

            # Apply mapping
            cluster_to_attack = dict(zip(cluster_mapping['Cluster'], cluster_mapping['Predicted_Attack_Type']))
            df_cyber_processed['Predicted_Attack_Type'] = df_cyber_processed['Cluster'].map(cluster_to_attack)

            # Save precomputed clusters for future use
            save_precomputed_clusters(df_cyber_processed, data_hash)

    # RESULT PREDICTIONS --> NEW version (works locally, is it the same online? Yes with the change of the name below)
        # Rename the column for display purposes
        df_cyber_processed = df_cyber_processed.rename(columns={'Predicted_Attack_Type': 'Predicted Attack Type'})

        # Define attack type mapping
        attack_mapping = {0: "DDoS", 1: "Intrusion", 2: "Malware"}

        # Create a display DataFrame without modifying the original data
        df_display = df_cyber_processed[['Cluster', 'Attack Type', 'Predicted Attack Type']].copy()

        # Convert attack type columns to integers (if they are stored as strings or floats)
        df_display['Attack Type'] = df_display['Attack Type'].astype(int, errors='ignore')
        df_display['Predicted Attack Type'] = df_display['Predicted Attack Type'].astype(int, errors='ignore')

        # Apply mapping (only for display)
        df_display['Attack Type'] = df_display['Attack Type'].map(attack_mapping)
        df_display['Predicted Attack Type'] = df_display['Predicted Attack Type'].map(attack_mapping)

        # Streamlit display of final predictions
        st.subheader("Predictions")
        st.dataframe(df_display.head(25))


    # RESULT PERFORMANCE METRICS
        # Model performance evaluation
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(df_cyber_processed[target], df_cyber_processed['Predicted Attack Type'])
        st.metric("Accuracy", f"{accuracy:.2%}")

        report_dict = classification_report(df_cyber_processed[target], df_cyber_processed['Predicted Attack Type'],
                                            output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        # Alert to say if we used precomputed clusters or if we have to recompute clusters
        message_placeholder = st.empty()

        if is_precomputed:
            message_placeholder.success("Using precomputed cluster results!")
        else:
            message_placeholder.warning("Computing clusters... This might take a few moments.")

        time.sleep(5)
        message_placeholder.empty()

        st.write("---")


    with tab4:
        st.title("HDBSCAN & XGBoost")

        # Generate a hash for the dataset
        data_hash = get_data_hash(df_cyber_processed)

        # Initialize train_cluster_labels to None
        train_cluster_labels = None
        X_train_top_features = None

        # Load precomputed clusters if they exist
        precomputed_clusters = load_precomputed_clusters(data_hash)

        if precomputed_clusters is not None:
            X_train_with_clusters = precomputed_clusters
            is_precomputed = True
        else:
            is_precomputed = False

            # Feature Selection with SelectKBest & f_classif
            best_features = SelectKBest(score_func=f_classif, k='all')
            best_features.fit(X_train, y_train)

            # Get Feature Scores
            feature_scores = best_features.scores_
            feature_names = X_train.columns

            # Create a DataFrame to visualize feature scores
            feature_scores_df = pd.DataFrame({
                'Feature': feature_names,
                'Score': feature_scores
            }).sort_values(by='Score', ascending=False)

            # Select Top Features
            top_features = feature_scores_df.head(10)['Feature'].tolist()
            X_train_top_features = X_train[top_features]
            X_val_top_features = X_val[top_features]
            X_test_top_features = X_test[top_features]

            # Clustering with HDBSCAN and PCA
            # Step 1: Select top features for clustering (using training data only)
            top_features = [f for f in feature_scores_df['Feature'].head(5).values if f != 'Attack Type']
            X_train_top_features = X_train[top_features]

            # Load HDBSCAN model from pickle file
            try:
                with open("WebApp/trained_models/best_hdbscan_model.pkl", "rb") as f:
                    hdbscan_model = pickle.load(f)
                    st.success("HDBSCAN model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading HDBSCAN model: {e}")
                # If loading fails, fit a new model
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=10,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )

            train_cluster_labels = hdbscan_model.fit_predict(X_train_top_features)

            # Add the new cluster labels to the training data
            X_train_with_clusters = X_train.copy()
            X_train_with_clusters['Cluster'] = train_cluster_labels

            # Save computed clusters
            save_precomputed_clusters(X_train_with_clusters, data_hash)

        # If the model was loaded successfully, use it to predict clusters
        if 'hdbscan_model' in locals():
            train_cluster_labels = hdbscan_model.fit_predict(X_train_top_features)


        # Step 4: Analyze the new cluster distribution
        if train_cluster_labels is not None:
            cluster_distribution = pd.Series(train_cluster_labels).value_counts()
            # Step 4: Analyze the new cluster distribution
            cluster_distribution = pd.Series(train_cluster_labels).value_counts()

            # Step 6: Visualize clusters in PCA-reduced dimensions
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_top_features)

            # Step 7: Prepare validation and test sets
            X_val_top_features = X_val[top_features]
            X_test_top_features = X_test[top_features]

            # Add placeholder cluster column for validation and test sets
            X_val_with_clusters = X_val.copy()
            X_test_with_clusters = X_test.copy()

            X_val_with_clusters['Cluster'] = -1  # Placeholder for validation
            X_test_with_clusters['Cluster'] = -1  # Placeholder for test


    # Cluster-Based XGBoost Classification with Label Encoding
        # Define selected features for XGBoost
        selected_features = [f for f in ['Anomaly Category', 'Time of Day', 'City', 'State'] if
                                 f in X_train.columns]

        cluster_models = {}
        cluster_predictions = []

        # Add the target column ('Attack Type') to the training data with clusters
        X_train_with_clusters['Attack Type'] = y_train

        # Iterate over unique clusters in the training dataset
        for cluster_id in X_train_with_clusters['Cluster'].unique():
            if cluster_id == -1:  # Skip noise cluster
                continue

            # Subset training data for the cluster
            cluster_train_data = X_train_with_clusters[X_train_with_clusters['Cluster'] == cluster_id]

            # Ensure selected features are available in the data
            cluster_features = [feature for feature in selected_features if feature in cluster_train_data.columns]

            # Split features and target for training
            X_cluster_train = cluster_train_data[cluster_features]
            y_cluster_train = cluster_train_data['Attack Type']  # Target column

            # Check if there is enough data in the cluster
            if len(X_cluster_train) < 8:
                print(f"Skipping cluster {cluster_id} due to insufficient data.")
                continue

            # Encode target labels
            label_encoder = LabelEncoder()
            y_cluster_train_encoded = label_encoder.fit_transform(y_cluster_train)

            # Train XGBoost model for this cluster
            model = XGBClassifier(eval_metric='logloss', random_state=42)
            model.fit(X_cluster_train, y_cluster_train_encoded)

            # Store the trained model and label encoder
            cluster_models[cluster_id] = {
                'model': model,
                'label_encoder': label_encoder
            }

            # Make predictions on the training data for this cluster
            cluster_preds = model.predict(X_cluster_train)

            # Decode predictions back to original labels
            cluster_preds_decoded = label_encoder.inverse_transform(cluster_preds)
            cluster_predictions.extend(zip(cluster_train_data.index, cluster_preds_decoded))


        # Store predictions as a DataFrame
        cluster_predictions_df = pd.DataFrame(cluster_predictions, columns=['Index', 'Predicted Attack Type'])
        cluster_predictions_df.set_index('Index', inplace=True)

        # Merge predictions back with the original dataset (if needed)
        X_train_with_predictions = X_train_with_clusters.copy()
        X_train_with_predictions['Predicted Attack Type'] = cluster_predictions_df['Predicted Attack Type']


    # RESULT PREDICTIONS
        # Define attack type mapping
        attack_mapping = {0: "DDoS", 1: "Intrusion", 2: "Malware"}

        # Create a display DataFrame without modifying the original data
        df_display = df_cyber_processed[['Cluster', 'Attack Type', 'Predicted Attack Type']].copy()

        # Convert attack type columns to integers (if they are stored as strings or floats)
        df_display['Attack Type'] = df_display['Attack Type'].astype(int, errors='ignore')
        df_display['Predicted Attack Type'] = df_display['Predicted Attack Type'].astype(int, errors='ignore')

        # Apply mapping (only for display)
        df_display['Attack Type'] = df_display['Attack Type'].map(attack_mapping)
        df_display['Predicted Attack Type'] = df_display['Predicted Attack Type'].map(attack_mapping)

        # Streamlit display of final predictions
        st.subheader("Predictions")
        st.dataframe(df_display.head(25))


    # RESULT: Evaluating Cluster-Based Classification Performance
        # Add predictions back to the DataFrame
        predictions_dict = dict(cluster_predictions)
        df_cyber_processed['Cluster_Predicted_Attack_Type'] = df_cyber_processed.index.map(predictions_dict)

        # Handle any NaN values in predictions
        df_cyber_processed['Cluster_Predicted_Attack_Type'].fillna(df_cyber_processed['Attack Type'].mode()[0],
                                                                       inplace=True)

        # Evaluate overall accuracy and classification report
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(df_cyber_processed['Attack Type'],
                                  df_cyber_processed['Cluster_Predicted_Attack_Type'])
        st.metric("Accuracy", f"{accuracy:.2%}")

        # Display classification report
        classification_report_output = classification_report(df_cyber_processed['Attack Type'],
                                                             df_cyber_processed['Cluster_Predicted_Attack_Type'],
                                                             output_dict=True)
        df_last_report = pd.DataFrame(classification_report_output).transpose()

        with st.expander("Classification Report"):
            st.dataframe(
                df_last_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        # Alert to say if we used precomputed clusters or if we have to recompute clusters
        message_placeholder = st.empty()

        if is_precomputed:
            message_placeholder.success("Using precomputed cluster results!")
        else:
            message_placeholder.warning("Computing clusters... This might take a few moments.")

        time.sleep(5)
        message_placeholder.empty()

        st.write("---")
