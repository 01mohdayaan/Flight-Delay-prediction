# Flight Delays Analysis & Prediction

## Project Overview
This project focuses on identifying patterns and predicting flight delays using Python-based data science techniques. By integrating flight schedules with weather condition data, the analysis explores how temporal factors and environmental conditions (like wind and precipitation) contribute to aviation delays.

## Author
* **Name:** Mohd Ayaan
* **Student ID:** 25262676

## Key Features
* **Data Integration:** Merged `flights_schedule.csv` and `flight_conditions_outcomes.csv` using unique flight identifiers.
* **Feature Engineering:** Developed custom features to improve analysis:
    * `dep_hour`: Extracted departure hour from timestamps.
    * `is_weekend`: Categorized flights based on the day of the week.
    * `delay_bucket`: Grouped delays into categories (e.g., *on_time*, *moderate_delay*, *severe_delay*).
    * `is_delayed`: Created a binary target variable for machine learning (delay ≥ 15 minutes).

## Prediction Feature Analysis
A core component of this project is the analysis of predictive features to determine their impact on flight punctuality. This section involves:

* **Correlation & Relationship Mapping:** Analyzing how independent variables (like weather severity) correlate with the target variable (`is_delayed`).
* **Temporal Analysis:** Evaluating the "Hour of Day" effect, identifying peak delay windows and how scheduling density affects performance.
* **Weather Impact Scoring:** Quantifying the influence of different precipitation levels (Dry vs. Heavy) and wind conditions (Calm vs. Very Windy) on departure lag.
* **Feature Selection:** Identifying and removing redundant or "leaky" features (such as `dep_delay`) to ensure the predictive model remains valid for real-world scenarios.

## Predictive Modeling: Logistic Regression
The project implements a **Logistic Regression** model to classify whether a flight will be delayed based on the engineered features.



### Implementation Details:
* **Binary Classification:** The model predicts the probability of the `is_delayed` class (True/False).
* **Data Splitting:** Utilized `train_test_split` to separate the data into training and validation sets, ensuring the model generalizes well to unseen flights.
* **Feature Scaling:** Applied standardization to numerical features to ensure the Logistic Regression algorithm converges efficiently and gives equal weight to all variables.
* **Evaluation Metrics:** The model's performance was assessed using:
    * **Accuracy Score:** To measure overall correct predictions.
    * **Confusion Matrix:** To visualize True Positives vs. False Positives, especially important given the class imbalance in aviation data.
    * **Classification Report:** Detailed Precision, Recall, and F1-score to understand how well the model identifies actual delays.

## Technologies Used
* **Language:** Python
* **Libraries:** * `Scikit-Learn`: For Logistic Regression, data splitting, and performance metrics.
    * `Pandas` & `NumPy`: Data manipulation and cleaning.
    * `Matplotlib` & `Seaborn`: Data visualization and trend analysis.

## Project Structure
1. **Task 1: Data Preparation:** Loading datasets, merging tables, and handling missing values.
2. **Feature Extraction:** Formatting datetime strings and creating categorical bins.
3. **Statistical Characterisation:** Summarizing the distribution of delays.
4. **Modeling & Evaluation:** Implementing Logistic Regression and analyzing the output metrics.

## Critical Challenges Addressed
* **Data Leakage:** Mitigated risks of including features that provide "future" information.
* **Class Imbalance:** Addressed the fact that on-time flights significantly outnumber delayed ones.
* **Preprocessing:** Resolved complex datetime formatting issues.

## Getting Started
### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
