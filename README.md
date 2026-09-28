🧠 Brain Tumor Detection System
A deep learning-based web application for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN). The project provides secure user authentication, AI-powered prediction, Grad-CAM explainability, PDF report generation, prediction history, and an analytics dashboard.
________________________________________
🚀 Features
## ✨ Features

### 🧠 AI-Powered Brain Tumor Detection
- Detects brain tumor classes from MRI images using a TensorFlow/Keras deep learning model.
- Provides the predicted tumor class along with a confidence score.
- Performs automated MRI image preprocessing before inference.

### 🔍 Explainable AI with Grad-CAM
- Generates Grad-CAM visualizations to highlight regions of the MRI image that influenced the model's prediction.
- Helps users understand the model's prediction rather than providing only a classification result.

### 👤 User Authentication
- User registration and login system.
- JWT-based authentication for securing API requests.
- User-specific prediction history.

### 📊 Prediction Dashboard
- Displays prediction statistics and analytics.
- Shows prediction history for authenticated users.
- Provides confidence and prediction information.

### 📄 Automated PDF Reports
- Generates a PDF report after prediction.
- Includes prediction results, confidence score, and Grad-CAM visualization.
- Reports can be viewed/downloaded from the application.

### ☁️ Cloud Storage
- Uses Supabase Storage for generated Grad-CAM images and PDF reports.
- Generates secure signed URLs for accessing stored files.

### 🗄️ Cloud Database
- Uses TiDB Cloud for persistent application data.
- Stores user information and prediction records.
- Stores prediction metadata, confidence scores, timestamps, and file paths.

### 🐳 Dockerized Application
- Separate Docker containers for the Streamlit frontend and FastAPI backend.
- Docker Compose manages the application services and internal network.
- Consistent deployment environment across development and production.

### ☁️ AWS EC2 Deployment
- Deployed on an Ubuntu AWS EC2 instance.
- Streamlit frontend runs on port `8501`.
- FastAPI backend runs on port `8000`.
- Containers are configured to automatically restart.

### 🔗 REST API Backend
- FastAPI-based REST API for frontend-backend communication.
- API endpoints for prediction, authentication, history, dashboard statistics, reports, and Grad-CAM results.

### 🧹 Temporary File Management
- Uploaded MRI images and generated intermediate files are temporarily processed on the server.
- Temporary files are cleaned up after prediction processing.
________________________________________
🎥 Full Project Demo

http://16.4.23.159:8501/

**▶️ Click the thumbnail above to watch the full project demo.**

### Demo Includes
•	🧠 Brain MRI image classification
•	🤖 TensorFlow deep learning model
•	⚡ FastAPI backend
•	📖 Swagger API documentation
•	🖥️ Streamlit frontend
•	🔐 User authentication
•	📊 Prediction and confidence score
•	🔥 Grad-CAM explainability
•	🗄 MySQL database integration
________________________________________
🛠 Technology Stack
## 🛠️ Technology Stack

### 🎨 Frontend
- **Streamlit** — Interactive web interface

### ⚙️ Backend
- **Python 3.11** — Core programming language
- **FastAPI** — REST API and backend framework
- **Uvicorn** — ASGI application server
- **Pydantic** — Data validation and API schemas
- **SQLAlchemy** — Database ORM

### 🧠 Machine Learning & AI
- **TensorFlow 2.16.1** — Deep learning framework
- **Keras 3.4.1** — Neural network/model interface
- **NumPy** — Numerical computation
- **OpenCV** — MRI image processing
- **Pillow (PIL)** — Image processing
- **Matplotlib** — Visualization
- **Grad-CAM** — Explainable AI / model visualization

### 🗄️ Database & Storage
- **TiDB Cloud** — Cloud database
- **MySQL / PyMySQL** — Database connectivity
- **Supabase Storage** — Cloud file storage
- **SQLAlchemy** — Database interaction

### 🔐 Authentication & Security
- **JWT (JSON Web Tokens)** — Authentication
- **Passlib + bcrypt** — Password hashing
- **Cryptography** — Security and encryption support
- **Python-dotenv** — Environment variable management

### 📄 Reporting
- **ReportLab** — PDF report generation

### 🐳 Containerization
- **Docker** — Application containerization
- **Docker Compose** — Multi-container orchestration
- **Docker Network** — Frontend/backend communication

### ☁️ Cloud & Deployment
- **AWS EC2** — Application hosting
- **Ubuntu Linux** — Server operating system
- **AWS Security Groups** — Network access control

### 🔧 Development & Version Control
- **Git** — Version control
- **GitHub** — Source code hosting and project management
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
 <img width="803" height="507" alt="image" src="https://github.com/user-attachments/assets/e79c8093-403c-4dc8-80d2-ce6eaa648e34" />

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

