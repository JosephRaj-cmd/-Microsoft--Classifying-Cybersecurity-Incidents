Project Overview
This project aims to enhance the efficiency of Security Operation Centers (SOCs) at Microsoft by developing a machine learning model capable of accurately classifying cybersecurity incidents. Using the comprehensive GUIDE dataset, the model predicts triage grades—True Positive (TP), Benign Positive (BP), or False Positive (FP)—based on historical evidence and customer responses. The goal is to provide SOC analysts with precise, context-rich recommendations to improve the security posture of enterprise environments.

Problem Statement
To support SOCs in managing and prioritizing cybersecurity incidents, this project focuses on training a robust model to predict incident grades. This helps in automating the triage process, thereby allowing SOC analysts to focus on critical threats. The model's performance is evaluated using the following metrics:

Macro-F1 Score
Precision
Recall
Business Use Cases
Security Operation Centers (SOCs): Automate the incident triage process to streamline analyst workflows.
Incident Response Automation: Enable guided response systems with accurate incident classifications.
Threat Intelligence: Leverage historical evidence for better threat detection.
Enterprise Security Management: Improve security by reducing false positives and ensuring real threats are prioritized.
Approach
Data Exploration and Preprocessing:
Analyze the dataset, handle missing values, and perform feature engineering.
Encode categorical variables and split data into training and validation sets with stratification.
Model Selection:
Start with a baseline model, then explore advanced models like Random Forests and Gradient Boosting Machines.
Use RandomizedSearchCV for hyperparameter tuning and cross-validation.
Model Evaluation:
Evaluate using macro-F1 score, precision, and recall.
Address class imbalance using techniques like SMOTE and adjusting class weights.
Final Testing:
Assess the model on the test dataset and compare results with the baseline.
Analyze feature importance and provide model interpretability.

Results
The project delivers a trained model that accurately classifies cybersecurity incidents with high macro-F1 score, precision, and recall. It includes a comprehensive analysis of the model's performance and insights into feature contributions, ensuring readiness for real-world deployment in SOC environments.

Dataset Overview
GUIDE Dataset: Contains evidence, alerts, and incidents related to cybersecurity events.
Target Variable: Incident triage grades—True Positive (TP), Benign Positive (BP), and False Positive (FP).
Size: The training dataset includes 1 million triage-annotated incidents, split into training (70%) and test (30%) sets.
Project Deliverables
Source Code: Documented code covering data preprocessing, model training, and evaluation.
Trained Model: Ready for deployment in SOC environments.
Documentation: Detailed report on the methodology, approach, and model performance.
Presentation: Summary of the project's objectives, challenges, and business impact.
