# DS Final Project

## 📚 Project Overview

This project is part of the "Introduction to Data Science" final course project.  
We work with a real-world dataset combining taxi trip records and weather conditions, and perform the following key tasks:

- Storing cleaned data into a MySQL database
- Designing a logical database schema
- Performing advanced feature engineering and preprocessing
- Building an automated data processing pipeline in Python
- Preparing datasets for downstream modeling tasks (classification, regression, etc.)

---

## 🏗️ Project Structure

```
DS_Final_Project/
🔁
👉 scripts/
👉👉 database_connection.py       # Database connection utility
👉👉 load_data.py                  # Loading tables into DataFrames
👉👉 preprocess.py                 # Data cleaning and preprocessing
👉👉 feature_engineering.py        # Advanced feature engineering
👉
👉 pipeline.py                       # Main pipeline script
👉 requirements.txt                  # Project dependencies
👉 README.md                         # Project documentation
```

---

## ⚙️ Requirements

- Python 3.8 or higher
- MySQL Server
- DBeaver (optional, for database management and queries)

Python libraries:

```
pandas
sqlalchemy
pymysql
scikit-learn
```

Install dependencies by running:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone or download the project repository.
2. Ensure your MySQL server is running.
3. Make sure the database `DS_Final_Project` exists and the required tables (`uber_trips`, `weather_data`, `taxi_zones`) are populated.
4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the full pipeline:

   ```bash
   python pipeline.py
   ```

This will automatically:

- Load data from MySQL
- Perform data preprocessing (cleaning, normalization)
- Perform feature engineering (creating new features)
- Save the processed datasets locally.

Processed output files:

- `processed_uber_trips.csv`
- `processed_weather_data.csv`
- `processed_taxi_zones.csv`

---

## 📸 Screenshots

(Include screenshots of your SQL queries, database schema, sample outputs, and successful pipeline execution here.)

---

## ✍️ Author

- Name: [Your Name Here]
- Course: Introduction to Data Science - Phase 2
- Year: 1403 (2025)

---

