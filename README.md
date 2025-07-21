# Emp_salary_prediction
This project predicts whether an employee earns more than $50K per year using demographic and work-related features from the Adult Census Income dataset. It combines data preprocessing, machine learning, and a user-friendly Streamlit web application for interactive salary classification.

🚀 Features
Predicts salary class (>50K or ≤50K) based on user input
Supports both single and batch (CSV) predictions
Encodes categorical features to match model training
Modern, easy-to-use Streamlit web interface
🛠️ System Requirements
Windows, macOS, or Linux
Python 3.8 or higher
4GB RAM (8GB recommended)
📦 Installation
Clone the repository:
bash
git clone https://github.com/yourusername/employee-salary-classification.git
cd employee-salary-classification
Install dependencies:
bash
pip install -r requirements.txt
Download or train the model:
Place 
best_model.pkl
 (trained model) in the project directory.
If you need to retrain, use the provided Jupyter notebook.
🏃‍♂️ Usage
Start the Streamlit app:
bash
streamlit run app.py
Single Prediction:
Enter employee details in the sidebar.
Click "Predict Salary Class" to see the result.
Batch Prediction:
Upload a CSV file with the required columns.
Download results with predicted classes.
📑 Project Structure
├── app.py                # Streamlit web app
├── Emp_salary_prediction.ipynb  # Model training notebook
├── best_model.pkl        # Trained ML model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
🔍 How it Works
Data Preprocessing: Cleans and encodes input features to match model training.
Model Prediction: Uses a Gradient Boosting Classifier for salary classification.
User Interface: Collects input, encodes features, and displays predictions.
🤝 Contributing
Pull requests and suggestions are welcome! Please open an issue to discuss changes.

📄 License
This project is licensed under the MIT License.
