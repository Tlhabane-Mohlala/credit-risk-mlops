# Credit Risk MLOps

End-to-end machine learning project for *credit default prediction* with a focus on *calibrated Probability of Default (PD)* estimation and production-oriented deployment.

## Project Overview
This project demonstrates how a credit risk model can move beyond notebook experimentation into a more production-ready workflow. The pipeline covers data preparation, model development, class imbalance handling, probability calibration, experiment tracking, API-based inference, and containerized deployment.

## Key Features
- Credit default prediction using supervised machine learning
- Class imbalance handling with *SMOTE*
- *Probability calibration* for more reliable PD estimates
- Model evaluation using:
  - Accuracy
  - ROC-AUC
  - Precision
  - Recall
  - F1-score
  - *Brier Score*
- Experiment tracking with *MLflow*
- Real-time inference using *FastAPI*
- Containerized deployment with *Docker*

## Why Calibration Matters
In credit risk, performance is not only about classification accuracy. Lending, pricing, and portfolio decisions depend on *reliable probability estimates. This project therefore emphasizes **calibration*, ensuring that predicted default probabilities are more aligned with real-world outcomes.

## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- MLflow
- FastAPI
- Postman
- Docker
- PostgreSQL
- Git / GitHub

## Project Structure
- model.py — model training, evaluation, and experiment logic
- db_connect.py — PostgreSQL connection logic
- requirements.txt — project dependencies
- data/ — input dataset(s)

## Current Status
- Model development completed
- Calibration workflow implemented
- API deployment tested locally
- Docker containerization completed
- GitHub workflow/CI to be finalized

## Project Goal
To build a production-oriented *credit risk MLOps pipeline* that combines predictive performance, calibration, explainability, and deployment readiness.

## Future Improvements
- Add CI workflow with GitHub Actions
- Add model monitoring and logging
- Deploy API to cloud infrastructure
- Extend explainability reporting

  # 1. Clone the repository
git clone https://github.com/Tlhabane-Mohlala/credit-risk-mlops.git

# 2. Navigate into the project
cd credit-risk-mlops

# 3. Create a virtual environment
python -m venv venv

# 4. Activate the environment
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the FastAPI app
uvicorn api:app --reload

## Author
*Matlhomola Mohlala*
