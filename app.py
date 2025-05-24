import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt


model = load_model('Model3.keras')

st.header('Stock Price Predictor')

st.subheader("CSV File Uploader and Data Processor")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    st.success("✅ File Uploaded Successfully!")
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Basic data processing options
    #st.write("### Basic Data Info:")
    #st.write(df.describe())
    #st.write("Number of rows:", df.shape[0])
    #st.write("Number of columns:", df.shape[1])

    # Optional: convert 'Date' column if exists
    if 'Date' in df.columns or 'DATE' in df.columns:
        date_col = 'Date' if 'Date' in df.columns else 'DATE'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        st.write(f"✅ Converted `{date_col}` to datetime and set as index.")


    df.dropna(inplace=True)

    # Insert a default placeholder
    column_options = ["-- Select Column --"] + list(df.columns)
    target_col = st.selectbox("Select target column for prediction", column_options)

    # Proceed only if the user has selected an actual column (not the placeholder)
    if target_col != "-- Select Column --":
        df_train = pd.DataFrame(df[target_col][0: int(len(df)*0.80)])
        df_test = pd.DataFrame(df[target_col][int(len(df)*0.8) : len(df)])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range = (0,1))  

        past_100_days= df_train.tail(100)
        data_test = pd.concat([past_100_days, df_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        x =[]
        y=[]

        for i in range(50, data_test_scale.shape[0]):
            x.append(data_test_scale[i-50:i])
            y.append(data_test_scale[i,0])  

        x,y = np.array(x), np.array(y)  

        y_predict = model.predict(x)

        y_predict_original = scaler.inverse_transform(y_predict)

        scale = 1/scaler.scale_
        y_predict = y_predict*scale
        y= y*scale


        plt.figure(figsize= (8,6))
        plt.plot(y_predict, 'r', label = 'predicted price')
        plt.plot(y, 'g', label='actual price')
        plt.legend()
        plt.show()  

        st.header('Actual price vs Predicted Price')
        st.pyplot(plt)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y, y_predict)
        mse = mean_squared_error(y, y_predict)

        st.write('MAE is: ', mae)
        st.write('MSE is: ', mse)

        y_predict = y_predict_original.flatten()

        # Create a full-length array for predicted prices filled with NaNs initially
        pred_full = np.full(shape=(len(df),), fill_value=np.nan)

        start_idx = len(df) - len(y_predict)  # 470 - 144 = 326
        pred_full[start_idx:] = y_predict

        df['Predicted'] = pred_full

        plt.figure(figsize=(8,6))
        plt.plot(df.index, df[target_col], label='Actual Prices', color='green')
        plt.plot(df.index, df['Predicted'], label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual vs Predicted Stock Prices')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


        



    else:
        st.warning("Please select a column to proceed.")

    
else:
    st.info("👈 Please upload a CSV file to get started.")



