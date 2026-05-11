import streamlit as st

# Page config MUST be first
st.set_page_config(layout="wide", page_title="Credit Scoring Model")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load and process data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("german_credit_data.csv")

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df = df.fillna("Unknown")

    # Encode categorical columns
    label_encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    # Convert specific columns to integers
    columns_to_convert = ['Age', 'Duration', 'Job', 'Credit amount']
    for col in columns_to_convert:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded, df, label_encoders

# Train multiple models
@st.cache_data
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label=1),
            'recall': recall_score(y_test, y_pred, pos_label=1),
            'f1': f1_score(y_test, y_pred, pos_label=1),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    return results

df_encoded, df_original, label_encoders = load_and_process_data()

# Features and target
X = df_encoded.drop("Risk", axis=1)
y = df_encoded["Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = train_models(X_train_scaled, y_train)

# Evaluate models
results = evaluate_models(models, X_test_scaled, y_test)

# Streamlit UI
st.title("🏦 Credit Scoring Model - Prediction System")

# Sidebar for model selection
st.sidebar.header("Model Selection & Evaluation")
selected_model = st.sidebar.selectbox("Choose Model:", list(models.keys()))

# Tab interface
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Predictions", "📈 Feature Analysis"])

# Tab 1: Model Comparison
with tab1:
    st.subheader("Model Performance Metrics")

    # Create metrics comparison table
    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    metrics_df = metrics_df.round(4)

    st.dataframe(metrics_df, use_container_width=True)

    # Visualize metrics
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(metrics_df[['accuracy', 'precision', 'recall', 'f1']])

    with col2:
        st.bar_chart(metrics_df['roc_auc'])

    # Best model
    best_model_name = metrics_df['f1'].idxmax()
    st.success(f"✓ Best Model (by F1-Score): **{best_model_name}** (F1: {metrics_df.loc[best_model_name, 'f1']:.4f})")

# Tab 2: Predictions
with tab2:
    st.subheader(f"Make Prediction with {selected_model}")

    # Display feature descriptions
    col1, col2, col3 = st.columns(3)

    feature_inputs = {}
    feature_cols = X.columns.tolist()

    # Create input fields for all features
    for idx, col in enumerate(feature_cols):
        if idx % 3 == 0:
            col_ref = col1
        elif idx % 3 == 1:
            col_ref = col2
        else:
            col_ref = col3

        with col_ref:
            if col in label_encoders:
                # Get unique values from original data
                unique_vals = df_original[col].unique()
                feature_inputs[col] = st.selectbox(f"{col}:", unique_vals)
            else:
                # Numeric input
                min_val = int(X[col].min())
                max_val = int(X[col].max())
                default_val = int((min_val + max_val) / 2)
                feature_inputs[col] = st.number_input(
                    f"{col} (Range: {min_val}-{max_val}):",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1
                )

    if st.button("🔮 Predict Credit Risk", key="predict_btn"):
        try:
            # Prepare input data
            input_dict = {}
            for col in feature_cols:
                if col in label_encoders:
                    input_dict[col] = label_encoders[col].transform([feature_inputs[col]])[0]
                else:
                    input_dict[col] = feature_inputs[col]

            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)

            # Get prediction
            selected_model_obj = models[selected_model]
            prediction = selected_model_obj.predict(input_scaled)[0]
            probability = selected_model_obj.predict_proba(input_scaled)[0]

            # Display results
            st.divider()
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                if prediction == 1:
                    st.success("✅ **GOOD CREDIT RISK** - Likely to repay")
                else:
                    st.error("❌ **BAD CREDIT RISK** - Higher default risk")

            with col_result2:
                st.metric("Confidence", f"{max(probability)*100:.2f}%")

            # Show probability breakdown
            st.subheader("Risk Probability Breakdown:")
            prob_df = pd.DataFrame({
                'Risk Category': ['Good Risk', 'Bad Risk'],
                'Probability': [probability[1]*100, probability[0]*100]
            })
            st.bar_chart(prob_df.set_index('Risk Category'))

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 3: Feature Analysis
with tab3:
    st.subheader("Feature Importance Analysis")

    # Get feature importance from Random Forest (if available)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        st.bar_chart(feature_importance.set_index('Feature'))

        st.dataframe(feature_importance.round(4), use_container_width=True)
