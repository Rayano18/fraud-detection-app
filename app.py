import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Set matplotlib style for better aesthetics
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Streamlit setup
st.set_page_config(page_title="Telecom Fraud Detection Dashboard", layout="wide")
st.title("Telecom Fraud Detection – Autoencoder Results")

# Cache functions for efficiency
@st.cache_resource
def load_autoencoder():
    try:
        return load_model("final_autoencoder_model.keras")
    except:
        try:
            return load_model("models/final_autoencoder_model.keras")
        except:
            st.warning("Model file not found. Please upload the model manually.")
            return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except:
        try:
            return joblib.load("models/scaler.pkl")
        except:
            st.warning("Scaler file not found. Scaled data must be provided.")
            return None

@st.cache_data
def load_threshold():
    try:
        with open("threshold.txt", "r") as f:
            return float(f.read())
    except:
        try:
            with open("models/threshold.txt", "r") as f:
                return float(f.read())
        except:
            return None

# Load model if available
model = load_autoencoder()
scaler = load_scaler()
threshold = load_threshold()

# Tab structure - CHANGED to just 2 tabs
tabs = st.tabs(["📊 Model Performance", "🧪 Test Dataset Analysis"])

with tabs[0]:
    st.header("Autoencoder Model Performance")
    
    # Model architecture visualization
    with st.expander("🏗️ Model Architecture"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Architecture Details")
            st.markdown("""
            | Layer | Units | Activation |
            | ----- | ----- | ---------- |
            | Input | 11 features | - |
            | Dense | 64 | ReLU |
            | Dropout | 0.2 | - |
            | Dense (Bottleneck) | 32 | ReLU |
            | Dense | 64 | ReLU |
            | Dropout | 0.2 | - |
            | Output | 11 features | Linear |
            """)
            
            # Add visual network diagram
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*44eDEuZBEsmG_TCAKRI3Kw.png", 
                     caption="Autoencoder Architecture Visualization",
                     width=400)
        
        with col2:
            st.subheader("Training Parameters")
            st.markdown("""
            - **Optimizer:** Adam (lr=0.0015)
            - **Loss Function:** Mean Squared Error (MSE)
            - **Batch Size:** 128
            - **L2 Regularization:** 1e-4
            - **Early Stopping:** Yes, patience=5
            - **Training Approach:** Fit on normal data, identify anomalies by reconstruction error
            """)
    
    # Upload training history
    with st.expander("📈 Training Performance"):
        history_file = st.file_uploader("Upload training history CSV (with loss and val_loss columns)", type=["csv"])
        
        if not history_file:
            # Show example chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            epochs = list(range(6))  # 0-5 epochs based on early stopping at epoch 5
            train_loss = [0.045, 0.024, 0.022, 0.019, 0.018, 0.017]
            val_loss = [0.044, 0.025, 0.023, 0.020, 0.019, 0.017]
            
            # Linear scale
            ax1.plot(epochs, train_loss, 'o-', label='Training Loss', color='#1f77b4')
            ax1.plot(epochs, val_loss, 'o-', label='Validation Loss', color='#ff7f0e')
            ax1.set_title('Loss Over Epochs (Best Model)')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Log scale
            ax2.semilogy(epochs, train_loss, 'o-', label='Training Loss', color='#1f77b4')
            ax2.semilogy(epochs, val_loss, 'o-', label='Validation Loss', color='#ff7f0e')
            ax2.set_title('Loss Over Epochs (Log Scale)')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss (log scale)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("👆 Upload your actual training history file for accurate visualization")
        else:
            try:
                history_df = pd.read_csv(history_file)
                if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
                    # Create matplotlib chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Linear scale
                    ax1.plot(history_df['loss'], 'o-', label='Training Loss', color='#1f77b4')
                    ax1.plot(history_df['val_loss'], 'o-', label='Validation Loss', color='#ff7f0e')
                    ax1.set_title('Training & Validation Loss')
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Loss (MSE)')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Log scale
                    ax2.semilogy(history_df['loss'], 'o-', label='Training Loss', color='#1f77b4')
                    ax2.semilogy(history_df['val_loss'], 'o-', label='Validation Loss', color='#ff7f0e')
                    ax2.set_title('Training & Validation Loss (Log Scale)')
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Loss (log scale)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Final performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Training Loss", f"{history_df['loss'].iloc[-1]:.6f}")
                    with col2:
                        st.metric("Final Validation Loss", f"{history_df['val_loss'].iloc[-1]:.6f}")
                    with col3:
                        ratio = history_df['val_loss'].iloc[-1] / history_df['loss'].iloc[-1]
                        st.metric("Validation/Training Ratio", f"{ratio:.2f}")
            except Exception as e:
                st.error(f"Error processing history file: {e}")

    # Train vs Test Error Comparison
    with st.expander("🔄 Train vs Test Error Comparison"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Training Error", "0.017843", 
                      help="Average reconstruction error on training data")
        with col2:
            st.metric("Mean Validation Error", "0.017349", 
                      help="Average reconstruction error on validation data")
        with col3:
            st.metric("Mean Test Error", "0.017498", 
                      help="Average reconstruction error on test data")
            
        st.success("The similar error values between training, validation, and test data (all around 0.017) indicate good generalization with appropriate regularization.")
    
    # Add footer with info
    st.markdown("---")
    st.markdown("""
    ### 📊 Telecom Fraud Detection Dashboard

    This dashboard visualizes the results of an autoencoder-based anomaly detection system for telecom data. 
    The autoencoder was trained to learn patterns from normal customer behavior and identify anomalies 
    by measuring reconstruction error.

    **Key metrics:**
    - Model Architecture: Autoencoder with L2 regularization and dropout layers
    - Training method: Unsupervised learning on normal behavior patterns
    - Detection method: Reconstruction error threshold (mean + 2×std)
    """)

with tabs[1]:
    st.header("🧪 Test Dataset Analysis")
    
    # Upload test dataset results
    uploaded_test = st.file_uploader("Upload test dataset results CSV", type=["csv"], key="test")
    
    if uploaded_test:
        df_test = pd.read_csv(uploaded_test)
        
        if 'cust_id' not in df_test.columns:
            st.error("CSV must contain 'cust_id' column.")
        else:
            # If model and scaler are available, allow live prediction
            if model is not None and scaler is not None and 'reconstruction_error' not in df_test.columns:
                st.subheader("🔮 Live Prediction")
                
                # Get features for prediction
                X = df_test.drop(columns=['cust_id'])
                
                # Scale and predict
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                error = np.mean(np.square(X_scaled - preds), axis=1)
                
                # Add to dataframe
                df_test['reconstruction_error'] = error
                
                st.success("✅ Successfully predicted reconstruction errors for test data!")
            
            # If reconstruction error is not there, stop
            if 'reconstruction_error' not in df_test.columns:
                st.error("CSV must contain 'reconstruction_error' column or model must be available for prediction.")
            else:
                # Calculate threshold if not provided
                if 'is_fraud' not in df_test.columns and threshold is not None:
                    df_test['is_fraud'] = (df_test['reconstruction_error'] > threshold).astype(int)
                    st.info(f"Applied threshold: {threshold:.6f}")
                elif 'is_fraud' not in df_test.columns:
                    # Use mean + 2*std
                    mean_error = df_test['reconstruction_error'].mean()
                    std_error = df_test['reconstruction_error'].std()
                    threshold = mean_error + 2 * std_error
                    st.info(f"Calculated threshold: {threshold:.6f} (mean + 2×std)")
                    df_test['is_fraud'] = (df_test['reconstruction_error'] > threshold).astype(int)
                
                df_test['fraud_label'] = df_test['is_fraud'].map({0: "Normal", 1: "Fraudulent"})
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Samples", f"{len(df_test):,}")
                with col2:
                    st.metric("Detected Anomalies", f"{df_test['is_fraud'].sum():,}")
                with col3:
                    fraud_pct = (df_test['is_fraud'].sum() / len(df_test)) * 100
                    st.metric("Anomaly Percentage", f"{fraud_pct:.2f}%")
                
                # Error distribution
                st.subheader("🔍 Test Set Error Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_test['reconstruction_error'], bins=100, kde=True, ax=ax)
                plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')
                plt.title('Reconstruction Error Distribution - Test Dataset')
                plt.xlabel('Reconstruction Error')
                plt.ylabel('Count')
                plt.legend()
                st.pyplot(fig)
                
                # ====== MOVED Feature Ratio Comparison ======
                # Select all numeric columns that aren't the reconstruction error or flags
                exclude_cols = ['cust_id', 'reconstruction_error', 'is_fraud', 'fraud_label']
                feature_cols = [col for col in df_test.columns if col not in exclude_cols and 
                               df_test[col].dtype in ['int64', 'float64', 'int32', 'float32']]
                
                if feature_cols:
                    # Calculate statistics for fraud comparison
                    st.subheader("📊 Feature Ratio Comparison (Fraud vs Normal)")
                    
                    # Calculate statistics
                    normal_df = df_test[df_test['is_fraud'] == 0]
                    fraud_df = df_test[df_test['is_fraud'] == 1]
                    
                    # Feature comparison with metrics
                    comparison_data = []
                    for feature in feature_cols:
                        normal_mean = normal_df[feature].mean()
                        fraud_mean = fraud_df[feature].mean()
                        normal_median = normal_df[feature].median()
                        fraud_median = fraud_df[feature].median()
                        
                        comparison_data.append({
                            'Feature': feature,
                            'Normal Mean': normal_mean,
                            'Fraud Mean': fraud_mean,
                            'Ratio (Fraud/Normal)': fraud_mean / normal_mean if normal_mean != 0 else float('inf'),
                            'Normal Median': normal_median,
                            'Fraud Median': fraud_median,
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df = comparison_df.sort_values('Ratio (Fraud/Normal)', ascending=False)
                    
                    # Create a more consolidated table by selecting only the most important columns
                    display_df = comparison_df[['Feature', 'Normal Mean', 'Fraud Mean', 'Ratio (Fraud/Normal)']]
                    
                    # Create interactive table with conditional formatting
                    def color_ratio(val):
                        if isinstance(val, (int, float)):
                            if val > 2:
                                return 'background-color: rgba(231, 76, 60, 0.2)'
                            elif val < 0.5:
                                return 'background-color: rgba(46, 204, 113, 0.2)'
                        return ''
                    
                    st.dataframe(display_df.style.applymap(color_ratio, subset=['Ratio (Fraud/Normal)']))
                    
                    # Bar chart of feature ratios - top features only
                    # Get top 10 features by absolute difference from 1.0
                    comparison_df['abs_diff_from_1'] = abs(comparison_df['Ratio (Fraud/Normal)'] - 1.0)
                    top_features = comparison_df.sort_values('abs_diff_from_1', ascending=False).head(10)
                    
                    # Create bar chart with matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot the ratios
                    bars = ax.bar(
                        top_features['Feature'],
                        top_features['Ratio (Fraud/Normal)'],
                        color=['firebrick' if x > 1 else 'forestgreen' for x in top_features['Ratio (Fraud/Normal)']]
                    )
                    
                    # Add a horizontal line at y=1
                    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}x',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points",
                                  ha='center', va='bottom')
                    
                    plt.title('Top Features by Fraud/Normal Ratio')
                    plt.xlabel('Feature')
                    plt.ylabel('Ratio (Fraud/Normal)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                # Test set feature analysis
                st.subheader("📊 Individual Feature Analysis")
                
                if feature_cols:
                    # Select visualization type
                    viz_type = st.radio(
                        "Select visualization type:",
                        ["Box Plot", "Violin Plot", "Bar Chart"],
                        horizontal=True,
                        key="test_viz_type"
                    )
                    
                    # Select a feature to analyze
                    selected_feature = st.selectbox(
                        "Select a feature to analyze:", 
                        options=feature_cols,
                        key="test_feature"
                    )
                    
                    if selected_feature:
                        if viz_type == "Box Plot":
                            # Box plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(
                                x="fraud_label", 
                                y=selected_feature,
                                data=df_test,
                                palette={"Normal": "#2ecc71", "Fraudulent": "#e74c3c"},
                                ax=ax
                            )
                            
                            # Add individual points
                            sns.stripplot(
                                x="fraud_label", 
                                y=selected_feature,
                                data=df_test,
                                size=4, 
                                alpha=0.3,
                                ax=ax,
                                palette={"Normal": "#2ecc71", "Fraudulent": "#e74c3c"}
                            )
                            
                            plt.title(f"{selected_feature} Distribution by Class")
                            plt.xlabel("Class")
                            plt.ylabel(selected_feature)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        elif viz_type == "Violin Plot":
                            # Violin plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.violinplot(
                                x="fraud_label", 
                                y=selected_feature,
                                data=df_test,
                                palette={"Normal": "#2ecc71", "Fraudulent": "#e74c3c"},
                                inner="box",
                                ax=ax
                            )
                            
                            plt.title(f"{selected_feature} Distribution by Class")
                            plt.xlabel("Class")
                            plt.ylabel(selected_feature)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        elif viz_type == "Bar Chart":
                            # Calculate statistics for normal and fraud groups
                            normal_stats = df_test[df_test['is_fraud'] == 0][selected_feature]
                            fraud_stats = df_test[df_test['is_fraud'] == 1][selected_feature]
                            
                            # Bar chart comparison
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Data to plot
                            labels = ["Normal", "Fraudulent"]
                            means = [normal_stats.mean(), fraud_stats.mean()]
                            medians = [normal_stats.median(), fraud_stats.median()]
                            
                            # Set width of bars
                            bar_width = 0.35
                            
                            # Set position of bar on X axis
                            r1 = np.arange(len(labels))
                            r2 = [x + bar_width for x in r1]
                            
                            # Make the plot
                            ax.bar(r1, means, color=['#2ecc71', '#e74c3c'], width=bar_width, label='Mean')
                            ax.bar(r2, medians, color=['#27ae60', '#c0392b'], width=bar_width, label='Median', alpha=0.7)
                            
                            # Add labels and title
                            plt.xlabel('Class')
                            plt.ylabel('Value')
                            plt.title(f"{selected_feature} - Mean and Median Comparison")
                            plt.xticks([r + bar_width/2 for r in r1], labels)
                            plt.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                # Show test set anomalies
                with st.expander("📋 View Test Set Anomalies"):
                    test_fraud = df_test[df_test['is_fraud'] == 1].sort_values('reconstruction_error', ascending=False)
                    st.dataframe(test_fraud)
                    
                    # Download button for test anomalies
                    csv_buffer = io.StringIO()
                    test_fraud.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="⬇️ Download Test Anomalies CSV",
                        data=csv_str,
                        file_name="test_anomalies.csv",
                        mime="text/csv"
                    )
                
                # Feature-level inspection for anomalies
                st.subheader("🔬 Feature-Level Anomaly Inspection")
                
                # Select a specific customer to analyze
                anomaly_ids = df_test[df_test['is_fraud'] == 1]['cust_id'].tolist()
                
                if anomaly_ids:
                    selected_option = st.selectbox(
                        "Select analysis option:",
                        ["Select specific customer", "Top anomaly", "Random anomaly"]
                    )
                    
                    selected_id = None
                    if selected_option == "Select specific customer":
                        selected_id = st.selectbox("Select customer ID:", anomaly_ids)
                    elif selected_option == "Top anomaly":
                        selected_id = df_test.loc[df_test['reconstruction_error'].idxmax(), 'cust_id']
                    elif selected_option == "Random anomaly":
                        selected_id = np.random.choice(anomaly_ids)
                    
                    if selected_id:
                            # Get customer data
                            customer = df_test[df_test['cust_id'] == selected_id].iloc[0]
                            
                            # Display customer details
                            st.subheader(f"Customer ID: {selected_id} (Reconstruction Error: {customer['reconstruction_error']:.6f})")
                            
                            # If we have the model, we can get per-feature reconstruction errors
                            if model is not None and scaler is not None:
                                # Calculate these averages once at the dataset level
                                if 'normal_avgs' not in st.session_state:
                                    st.session_state.normal_avgs = df_test[df_test['is_fraud'] == 0][feature_cols].mean()
                                    st.session_state.fraud_avgs = df_test[df_test['is_fraud'] == 1][feature_cols].mean()
                                    st.write("Debug: Averages calculated and stored")
                                else:
                                    st.write("Debug: Using stored averages")
                                
                                # Get features
                                X_customer = df_test[df_test['cust_id'] == selected_id][feature_cols].values
                                
                                # Scale
                                X_customer_scaled = scaler.transform(X_customer)
                                
                                # Get reconstruction
                                X_customer_recon = model.predict(X_customer_scaled)
                                
                                # Calculate per-feature errors
                                feature_errors = np.square(X_customer_scaled - X_customer_recon)[0]
                                
                                # Use the stored averages
                                normal_avgs = st.session_state.normal_avgs
                                fraud_avgs = st.session_state.fraud_avgs
                            
                            # Create comprehensive feature analysis DataFrame (combining the two tables)
                            comprehensive_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Customer Value': X_customer[0],
                                'Normal Average': [normal_avgs[col] for col in feature_cols],
                                'Anomaly Average': [fraud_avgs[col] for col in feature_cols],
                                'Cust/Normal Ratio': [X_customer[0][i] / normal_avgs[col] if normal_avgs[col] != 0 else float('inf') 
                                                   for i, col in enumerate(feature_cols)],
                                'Recon Error': feature_errors
                            })
                            
                            # Sort by reconstruction error by default
                            comprehensive_df = comprehensive_df.sort_values('Recon Error', ascending=False)
                            
                            # Add conditional formatting
                            def highlight_anomalies(s):
                                is_ratio = 'Ratio' in s.name
                                is_error = 'Error' in s.name
                                
                                if is_ratio:
                                    return ['background-color: rgba(231, 76, 60, 0.2)' if v > 2 else 
                                            'background-color: rgba(46, 204, 113, 0.2)' if v < 0.5 else 
                                            '' for v in s]
                                elif is_error:
                                    normalized_errors = (s - s.min()) / (s.max() - s.min() + 1e-10)
                                    return ['background-color: rgba(231, 76, 60, {})'.format(min(v*0.8 + 0.2, 0.8)) 
                                            for v in normalized_errors]
                                else:
                                    return ['' for _ in s]
                            
                            # Show consolidated table with styling
                            st.subheader("Comprehensive Feature Analysis")
                            st.dataframe(comprehensive_df.style.apply(highlight_anomalies))
                            
                            # Visualization of top contributors to anomaly
                            st.subheader("Top Features Contributing to Anomaly")
                            
                            # Get top 8 features by reconstruction error
                            top_error_features = comprehensive_df.head(8)
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            bars = ax.bar(
                                top_error_features['Feature'],
                                top_error_features['Recon Error'],
                                color=plt.cm.Reds(np.linspace(0.5, 0.9, len(top_error_features)))
                            )
                            
                            # Add value labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.annotate(f'{height:.6f}',
                                          xy=(bar.get_x() + bar.get_width() / 2, height),
                                          xytext=(0, 3),  # 3 points vertical offset
                                          textcoords="offset points",
                                          ha='center', va='bottom',
                                          fontsize=8, rotation=45)
                            
                            plt.title("Feature-Level Reconstruction Errors")
                            plt.xlabel("Feature")
                            plt.ylabel("Reconstruction Error")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                
                # Allow downloading entire test results
                csv_buffer = io.StringIO()
                df_test.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                st.download_button(
                    label="⬇️ Download Complete Test Results",
                    data=csv_str,
                    file_name="test_results.csv",
                    mime="text/csv"
                )