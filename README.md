# Flight Delays Analysis & Predictive Modeling

## Project Overview
This project provides a comprehensive analysis of flight delays by integrating flight schedule data with weather conditions. It covers the full data science pipeline, including data cleaning, feature engineering, exploratory data analysis (EDA), and the implementation of multiple machine learning classifiers to predict whether a flight will be delayed.

## Author
* **Name:** Mohd Ayaan
* **Student ID:** 25262676
* **Academic Year:** Autumn 2025/26

## Project Structure
The notebook is organized into three core tasks:
1.  **Data Preparation and Characterisation:** Merging datasets, cleaning missing values, and engineering new features.
2.  **Classification and Evaluation:** Training and comparing Logistic Regression, Decision Tree, and Random Forest models.
3.  **Predictive Feature Analysis:** Analyzing the impact of distance, time, and weather on delay outcomes.

## Key Features & Methodology

### 1. Data Integration & Engineering
* **Dataset Merging:** Combined `flights_schedule.csv` and `flight_conditions_outcomes.csv` using the unique `flight_id`.
* **Feature Engineering:**
    * **Temporal Features:** Extracted `dep_hour` and `day_of_week`; created a `weekend` flag.
    * **Distance Categorization:** Binned flights into `short` (<300km), `medium` (300-800km), and `long` (>=800km) categories.
    * **Weather Severity Score:** Created a composite numerical score by mapping and summing categorical precipitation and wind conditions.
* **Target Variable:** Defined a binary `delayed` flag (True if delay ≥ 15 minutes).

### 2. Predictive Modeling & Logistic Regression
The project evaluates three machine learning algorithms to classify flights as delayed or on-time:
* **Logistic Regression:** Used as the primary baseline for binary classification.
* **Decision Tree Classifier:** Implemented to capture non-linear relationships.
* **Random Forest Classifier:** An ensemble method used to improve accuracy and reduce overfitting.

**Evaluation Metrics:**
Models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**. A **Confusion Matrix** was generated for each to analyze how well the models handled the class imbalance (where on-time flights significantly outnumber delayed ones).

### 3. Predictive Feature Analysis
This section investigates the "why" behind the delays:
* **Weather Impact:** Visualized the correlation between the composite `weather_severity` score and the likelihood of a delay.
* **Distance Analysis:** Compared delay proportions across different flight distance categories.
* **Temporal Trends:** Identified peak delay hours by analyzing the mean delay rate relative to the hour of departure.

## Technologies Used
* **Language:** Python
* **Data Handling:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-Learn`

## How to Use
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/flight-delay-analysis.git](https://github.com/YOUR_USERNAME/flight-delay-analysis.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Data Files:** Ensure `flights_schedule.csv` and `flight_conditions_outcomes.csv` are in the project folder.
4.  **Run the Notebook:** Execute all cells in `assignment2_flight_delays.ipynb`.

## Conclusions
The analysis demonstrates that flight delays are strongly influenced by weather severity and departure timing. While the **Random Forest** and **Logistic Regression** models provide a strong foundation for prediction, future improvements could involve using **SMOTE** for better class balance or experimenting with **XGBoost** for higher precision.
