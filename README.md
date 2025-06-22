🚔 Crime Rate Prediction Dashboard
A machine learning web app built using Streamlit that predicts the crime rate for Indian cities based on inputs like city, crime type, year, and population. The model is trained using the XGBoost Regressor.

📌 Features
🔢 Predict crime rate using machine learning (XGBoost)

📉 Visualize trends in crime rate over the years by city and crime type

📊 View performance metrics like MAE, RMSE, and R² Score

✅ Built with Streamlit for an interactive UI

📁 Uses a real-world dataset (crime_data.xlsx)

📂 Project Structure
bash
Copy
Edit
crime-rate-predictor/
├── app.py                  # Main Streamlit app
├── crime_data.xlsx         # Dataset used for training
├── README.md               # Project description
└── requirements.txt        # Python dependencies
🧠 Machine Learning Model
Model Used: XGBoost Regressor

Type: Regression

Input Features:

Year

City (Encoded)

Population

Crime Type (Encoded)

Target: Crime Rate (Numerical)

📈 Model Evaluation
Metric	Value
📊 MAE (Mean Abs Error)	0.92
📉 RMSE	1.58
📈 R² Score	0.99

✅ Note: Accuracy is not applicable here because this is a regression problem, not classification.

▶️ How to Run the App
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/crime-rate-predictor.git
cd crime-rate-predictor
Install the requirements

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
📄 Dataset Details
Dataset Name: crime_data.xlsx

Fields: Year, City, Population, Type, Crime Rate

Cities include: Delhi, Mumbai, Chennai, etc.

Crime types: Murder, Theft, Assault, etc.

📚 Research Background
This project is inspired by studies analyzing crime patterns in Indian metro cities using machine learning. It helps forecast potential crime rates, assisting policy makers in decision-making.

🚀 Future Improvements
Add classification model to predict risk level (High/Medium/Low)

Include more features like education level, unemployment rate

Deploy on cloud (Streamlit Cloud / Heroku)

🙌 Acknowledgements
XGBoost library

Streamlit framework

Public crime datasets from government portals
