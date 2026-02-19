# Flight Delays Analysis & Predictive Modeling

## Project Overview
This project provides a data-driven analysis of flight delays by integrating flight schedules with weather condition datasets. The goal is to identify key factors that lead to departure delays and to build a predictive model that can classify whether a flight will be delayed based on environmental and temporal features.

## Author
* **Name:** Mohd Ayaan
* **Student ID:** 25262676
* **Academic Year:** Autumn 2025/26

## Project Structure
The analysis follows a structured data science workflow divided into three main tasks:
1.  **Data Preparation and Characterisation:** Merging datasets, handling missing values, and performing initial statistical summaries.
2.  **Classification and Evaluation:** Building and testing multiple machine learning models to predict delay outcomes.
3.  **Predictive Feature Analysis:** Diving deep into the specific features (like weather and time) that drive delays.

## Key Features & Methodology

### 1. Data Integration & Engineering
* **Dataset Merging:** Joined `flights_schedule.csv` and `flight_conditions_outcomes.csv` via the unique `flight_id`.
* **Feature Engineering:** Developed custom features to enhance model performance:
    * `dep_hour`: Extracted from timestamps to analyze peak delay times.
    * `is_weekend`: Categorized flights based on the day of the week.
    * `weather_severity_score`: A composite numerical metric derived from precipitation and wind conditions.
    * `distance_cat`: Binned flights into Short, Medium, and Long-haul categories.
* **Target Variable:** A binary `delayed` flag representing departures delayed by 15 minutes or more.

### 2. Predictive Modeling: Logistic Regression & Ensembles
The project evaluates three distinct machine learning algorithms:
* **Logistic Regression:** Used as the baseline model for binary classification.
* **Decision Tree Classifier:** Implemented to capture non-linear relationships and interactions between features.
* **Random Forest Classifier:** An ensemble approach used to improve accuracy and generalizability while reducing overfitting.

**Evaluation:** Models were compared using **Accuracy**, **Precision**, **Recall**, and **F1-Score**, with a heavy focus on the **Confusion Matrix** to ensure the model successfully identifies delays despite the class imbalance.

### 3. Predictive Feature Analysis
* **Weather Severity:** Quantified the direct link between high precipitation/wind levels and increased delay rates.
* **Temporal Trends:** Identified specific hours of the day where congestion leads to higher probabilities of delay.
* **Distance Correlation:** Analyzed if flight length correlates with punctuality.

## Technologies Used
* **Python:** Primary programming language.
* **Pandas & NumPy:** Data manipulation and cleanup.
* **Scikit-Learn:** Machine learning modeling and evaluation.
* **Matplotlib & Seaborn:** Data visualization.

## Getting Started
### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
