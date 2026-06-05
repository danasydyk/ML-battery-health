import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Battery Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to force clean baby blue and green aesthetics and style components
st.markdown(
    """
    <style>
    /* Main body background color and text color */
    .stApp {
        background-color: #eef7fc;
        color: #1a303a;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #d0ebf7;
    }
    section[data-testid="stSidebar"] .write {
        color: #1a303a;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1a303a;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Custom container for output card */
    .prediction-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #27ae60;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: #27ae60;
        margin: 10px 0;
    }
    
    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #27ae60;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #219653;
        color: white;
        border: none;
    }
    
    /* Info/Success/Warning blocks override */
    .stAlert {
        background-color: #ffffff;
        color: #1a303a;
        border: 1px solid #d0ebf7;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please run train_model.py first to train the model.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("Smartphone Battery Health Predictor")
st.markdown(
    "Use this tool to predict current battery health percentage based on usage habits and device characteristics. "
    "Predictions are powered by a Gradient Boosting Regressor model trained on device telemetry data."
)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Single Device Prediction", "Batch Prediction"]
)

# Model details in sidebar
st.sidebar.markdown("---")
st.sidebar.title("Model Details")
st.sidebar.write("Model type: Gradient Boosting Regressor")
st.sidebar.write("Evaluation score: 0.94 R2 Score")
st.sidebar.write("Target: Current Battery Health Percentage")

# 11 columns in the exact order required by the preprocessor
feature_columns = [
    "device_age_months",
    "battery_capacity_mah",
    "avg_screen_on_hours_per_day",
    "avg_battery_temp_celsius",
    "fast_charging_usage_percent",
    "overnight_charging_freq_per_week",
    "gaming_hours_per_week",
    "video_streaming_hours_per_week",
    "background_app_usage_level",
    "signal_strength_avg",
    "usage_intensity_score"
]

if app_mode == "Single Device Prediction":
    st.header("Single Device Prediction")
    st.write("Adjust the inputs below to predict the battery health for a specific device.")
    
    # Form layout with 3 columns for numeric values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        device_age = st.slider(
            "Device Age (Months)",
            min_value=0,
            max_value=48,
            value=24,
            help="Number of months the device has been in use."
        )
        
        battery_capacity = st.slider(
            "Battery Capacity (mAh)",
            min_value=3000,
            max_value=5000,
            value=4000,
            step=100,
            help="Design capacity of the smartphone battery."
        )
        
        screen_on_time = st.slider(
            "Average Screen On Time (Hours/Day)",
            min_value=1.0,
            max_value=12.0,
            value=5.5,
            step=0.1,
            help="Average daily screen on time."
        )
        
    with col2:
        battery_temp = st.slider(
            "Average Battery Temperature (Celsius)",
            min_value=20.0,
            max_value=45.0,
            value=32.0,
            step=0.1,
            help="Average battery temperature during use."
        )
        
        fast_charging_pct = st.slider(
            "Fast Charging Usage (Percent)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Percentage of total charge cycles that used fast charging."
        )
        
        overnight_freq = st.slider(
            "Overnight Charging Frequency (Times/Week)",
            min_value=0,
            max_value=7,
            value=3,
            help="How many times per week the phone is left to charge overnight."
        )
        
    with col3:
        gaming_hours = st.slider(
            "Gaming Time (Hours/Week)",
            min_value=0.0,
            max_value=25.0,
            value=4.0,
            step=0.1,
            help="Total hours spent gaming per week."
        )
        
        streaming_hours = st.slider(
            "Video Streaming (Hours/Week)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="Total hours spent streaming video per week."
        )
        
        usage_intensity = st.slider(
            "Usage Intensity Score",
            min_value=5.0,
            max_value=10.0,
            value=10.0,
            step=0.1,
            help="Calculated score reflecting device usage intensity."
        )
        
    # Categorical variables in a separate row
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        bg_app_level = st.selectbox(
            "Background App Usage Level",
            ["Low", "Medium", "High"],
            index=1,
            help="Intensity of background app activity."
        )
        
    with col_cat2:
        signal_strength = st.selectbox(
            "Average Signal Strength",
            ["Poor", "Moderate", "Good"],
            index=2,
            help="Average network signal quality."
        )
        
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "device_age_months": device_age,
        "battery_capacity_mah": battery_capacity,
        "avg_screen_on_hours_per_day": screen_on_time,
        "avg_battery_temp_celsius": battery_temp,
        "fast_charging_usage_percent": fast_charging_pct,
        "overnight_charging_freq_per_week": overnight_freq,
        "gaming_hours_per_week": gaming_hours,
        "video_streaming_hours_per_week": streaming_hours,
        "background_app_usage_level": bg_app_level,
        "signal_strength_avg": signal_strength,
        "usage_intensity_score": usage_intensity
    }])
    
    # Ensure correct column order
    input_data = input_data[feature_columns]
    
    # Run prediction
    if model is not None:
        prediction = model.predict(input_data)[0]
        
        # Display results in a structured container
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Determine health status description based on predicted value
        if prediction >= 80:
            status = "Good Health"
            color = "#27ae60"
        elif prediction >= 50:
            status = "Moderate Health - Consider monitoring battery temperature"
            color = "#f39c12"
        else:
            status = "Poor Health - Battery replacement recommended"
            color = "#c0392b"
            
        st.markdown(
            f"""
            <div class="prediction-card">
                <div>Predicted Current Battery Health:</div>
                <div class="prediction-value">{prediction:.2f}%</div>
                <div style="font-weight: bold; color: {color};">{status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif app_mode == "Batch Prediction":
    st.header("Batch Prediction")
    st.write(
        "Upload a CSV file containing device features to generate battery health predictions for multiple devices."
    )
    
    # Template download help
    with st.expander("CSV Schema Requirement"):
        st.write("The uploaded CSV file must contain the following columns:")
        st.code(", ".join(feature_columns))
        st.write("Optional column: Device_ID (retained in output to map predictions back to specific devices).")
        
        # Generate a small sample template
        sample_df = pd.DataFrame([{
            "Device_ID": "sample-device-123",
            "device_age_months": 12,
            "battery_capacity_mah": 4000,
            "avg_screen_on_hours_per_day": 5.0,
            "avg_battery_temp_celsius": 30.5,
            "fast_charging_usage_percent": 25.0,
            "overnight_charging_freq_per_week": 4,
            "gaming_hours_per_week": 2.5,
            "video_streaming_hours_per_week": 8.0,
            "background_app_usage_level": "Medium",
            "signal_strength_avg": "Good",
            "usage_intensity_score": 10.0
        }])
        st.write("Sample row structure:")
        st.dataframe(sample_df)
        
        csv_template = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="battery_features_template.csv",
            mime="text/csv"
        )
        
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )
    
    if uploaded_file is not None and model is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Check for missing columns
            missing_cols = [col for col in feature_columns if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
            else:
                # Prepare data for prediction
                prediction_input = input_df[feature_columns]
                
                # Run prediction
                predictions = model.predict(prediction_input)
                
                # Build result DataFrame
                result_df = input_df.copy()
                result_df["predicted_battery_health_percent"] = predictions
                
                # Reorder to put prediction near ID or at start
                cols = list(result_df.columns)
                if "Device_ID" in cols:
                    cols.remove("Device_ID")
                    cols = ["Device_ID", "predicted_battery_health_percent"] + cols
                else:
                    cols = ["predicted_battery_health_percent"] + cols
                result_df = result_df[cols]
                
                st.success("Predictions generated successfully!")
                
                # Show summary metrics
                st.subheader("Batch Summary")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Devices", len(result_df))
                col_m2.metric("Average Predicted Health", f"{predictions.mean():.2f}%")
                col_m3.metric("Minimum Predicted Health", f"{predictions.min():.2f}%")
                
                # Show results table
                st.subheader("Prediction Results")
                st.dataframe(result_df)
                
                # Download button for result
                result_csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=result_csv,
                    file_name="battery_health_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
