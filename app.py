import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel('crime_data.xlsx')
    return df

# Preprocess dataset
@st.cache_data
def preprocess_data(df):
    le_city = LabelEncoder()
    df['City_enc'] = le_city.fit_transform(df['City'])

    le_type = LabelEncoder()
    df['Type_enc'] = le_type.fit_transform(df['Type'])

    X = df[['Year', 'City_enc', 'Population (in Lakhs) (2011)+', 'Type_enc']]
    y = df['Crime Rate']
    return X, y, le_city, le_type

# Train model
@st.cache_resource
def train_model(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model

# Load data
df = load_data()

st.title("🚔 Crime Rate Prediction Dashboard (Indian Cities)")

# Show full raw data option
if st.checkbox("Show full raw data"):
    st.write(df)

# Preprocess data
X, y, le_city, le_type = preprocess_data(df)

# Train model
model = train_model(X, y)

# Sidebar inputs for prediction
st.sidebar.header("Enter Crime Details for Prediction")

year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2023)
city = st.sidebar.selectbox("City", options=df['City'].unique())
population = st.sidebar.number_input("Population (in Lakhs)", min_value=0.0, max_value=1000.0, value=10.0, step=0.01)
crime_type = st.sidebar.selectbox("Crime Type", options=df['Type'].unique())

if st.sidebar.button("Predict Crime Rate"):
    # Encode inputs
    try:
        city_enc = le_city.transform([city])[0]
        type_enc = le_type.transform([crime_type])[0]

        input_df = pd.DataFrame([[year, city_enc, population, type_enc]],
                                columns=['Year', 'City_enc', 'Population (in Lakhs) (2011)+', 'Type_enc'])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Crime Rate: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Visualization section: Filter by city & crime type
st.subheader("Crime Rate Trends by City and Crime Type")

city_to_plot = st.selectbox("Select city to visualize", df['City'].unique(), key='city_plot')
crime_type_to_plot = st.selectbox("Select crime type to visualize", df['Type'].unique(), key='type_plot')

filtered_data = df[(df['City'] == city_to_plot) & (df['Type'] == crime_type_to_plot)]

if filtered_data.empty:
    st.warning("No data available for this city and crime type combination.")
else:
    if st.checkbox("Show filtered data", key='show_filtered_data'):
        st.write(filtered_data)

    chart_data = filtered_data.groupby('Year')['Crime Rate'].mean().reset_index()
    st.line_chart(data=chart_data, x='Year', y='Crime Rate')
