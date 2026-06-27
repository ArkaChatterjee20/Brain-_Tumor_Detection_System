🧠 Brain Tumor Detection System
A deep learning-based web application for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN). The project provides secure user authentication, AI-powered prediction, Grad-CAM explainability, PDF report generation, prediction history, and an analytics dashboard.
________________________________________
🚀 Features
•	🧠 CNN-based Brain Tumor Detection
•	⚡ FastAPI REST API
•	🔐 JWT Authentication
•	🗄 MySQL Database Integration
•	🔥 Grad-CAM Explainability
•	📄 Automatic PDF Report Generation
•	📜 Prediction History
•	📊 Dashboard Analytics
•	👨‍💼 Admin Dashboard
•	🌐 Streamlit Frontend
•	📂 Image Upload Support (JPG, JPEG, PNG)
________________________________________
🏗 Project Architecture
                    Streamlit Frontend
                           │
                           │ REST API
                           ▼
                    FastAPI Backend
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      CNN Model       MySQL Database    PDF Reports
          │
          ▼
      Grad-CAM Heatmap
________________________________________
🛠 Technology Stack
Frontend
•	Streamlit
Backend
•	FastAPI
•	Uvicorn
Machine Learning
•	TensorFlow
•	Keras
•	OpenCV
•	NumPy
Database
•	MySQL
•	SQLAlchemy
Authentication
•	JWT (JSON Web Token)
Visualization
•	Grad-CAM
Report Generation
•	ReportLab
________________________________________
📁 Project Structure
Brain-Tumor-Detection-System
│
├── auth/
├── backend/
├── database/
├── frontend/
├── ml/
├── models/
├── pages/
├── styles/
├── requirements.txt
├── .gitignore
└── README.md
________________________________________
🧠 Brain Tumor Classes
•	Glioma
•	Meningioma
•	Pituitary Tumor
•	No Tumor
________________________________________
📥 Installation
1. Clone the repository
git clone https://github.com/ArkaChatterjee20/Brain-_Tumor_Detection_System.git

cd Brain-_Tumor_Detection_System
2. Create a virtual environment
python -m venv .venv
Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
________________________________________
⚙ Configure Environment Variables
Create a .env file in the project root.
Example:
SECRET_KEY=your_secret_key

DATABASE_URL=mysql+pymysql://username:password@localhost/database_name

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=60
________________________________________
▶ Run FastAPI Backend
uvicorn backend.main:app --reload
Backend URL:
http://127.0.0.1:8000
API Documentation:
http://127.0.0.1:8000/docs
________________________________________
▶ Run Streamlit Frontend
streamlit run frontend/streamlit_app.py
Frontend URL:
http://localhost:8501
________________________________________
📊 Application Workflow
1.	Register/Login
2.	Upload MRI Image
3.	CNN predicts tumor class
4.	Generate Grad-CAM heatmap
5.	Generate PDF report
6.	Save prediction in MySQL
7.	Display prediction history
8.	View dashboard analytics
________________________________________
📷 Screenshots
Home Page
 <img width="749" height="416" alt="image" src="https://github.com/user-attachments/assets/9e714ffc-e6c6-4413-80ed-497ba82868b6" />

________________________________________
Login Page
 <img width="759" height="314" alt="image" src="https://github.com/user-attachments/assets/d6635841-ab35-445e-a00b-20d9eee6bb63" />

________________________________________
Prediction Result
 <img width="803" height="507" alt="image" src="https://github.com/user-attachments/assets/e79c8093-403c-4dc3-80d2-ce6eaa648e34" />

________________________________________
Grad-CAM
 <img width="751" height="748" alt="image" src="https://github.com/user-attachments/assets/70309fbb-3ca7-4e1a-9441-3a3f03dba28f" />

________________________________________
Prediction History
 <img width="827" height="406" alt="image" src="https://github.com/user-attachments/assets/f879cdc8-2e54-44dd-b905-44eb542940fc" />

________________________________________
Dashboard
 
 <img width="789" height="541" alt="image" src="https://github.com/user-attachments/assets/4a507b4b-b113-45b8-b245-6746e5c3380f" />
 <img width="786" height="345" alt="image" src="https://github.com/user-attachments/assets/1dc4fa05-e8a6-4752-a3ee-4034e4bb7aaa" />


________________________________________
Admin Dashboard
 <img width="766" height="725" alt="image" src="https://github.com/user-attachments/assets/d1879a6e-c3f0-4813-b9e4-a8763d04f30c" />

 
________________________________________
📚 Dataset
This project uses a public Brain MRI dataset for training.
Example source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
The dataset is not included in this repository because GitHub limits files larger than 100 MB.
________________________________________
🌍 Deployment
Backend
To be added after deployment.
Example:
https://your-backend.onrender.com
________________________________________
Frontend
To be added after deployment.
Example:
https://your-streamlit-app.streamlit.app
________________________________________
🔮 Future Improvements
•	Docker Support
•	CI/CD Pipeline using GitHub Actions
•	Role-Based Authentication
•	Email Notifications
•	Multi-language Support
•	Model Retraining Pipeline
•	Explainable AI Improvements
________________________________________
👨‍💻 Author
Arka Chatterjee
GitHub:
https://github.com/ArkaChatterjee20
________________________________________
📄 License
This project is intended for educational and research purposes.

