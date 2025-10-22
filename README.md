# Ride Sharing Analytics Using Spark Streaming and MLlib

The objective of this project is to extend the real-time data analytics pipeline for a ride-sharing service by integrating Machine Learning capabilities using Apache Spark's MLlib. The pipeline now includes offline model training and real-time fare prediction and trend forecasting.

## **Prerequisites**

Before starting the assignment, ensure you have the following software installed and properly configured on your machine:

1. **Python 3.x**:
   - [Download and Install Python](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python3 --version
     ```

2. **PySpark**:
   - Install using `pip`:
     ```bash
     pip install pyspark
     ```

3. **Faker**:
   - Install using `pip`:
     ```bash
     pip install faker
     ```

4. **Java 8+**:
   - Verify installation:
     ```bash
     java -version
     ```

## Repository Structure

The project repository should have the following structure:

```
ride-sharing-analytics/
├── models/
│   ├── fare_model/
│   │   ├── data/
│   │   └── metadata/
│   └── fare_trend_model_v2/
│       ├── data/
│       └── metadata/
├── OutputScreenshots/
│   ├── task4_screenshot.png
│   └── task5_screenshot.png
├── task4.py
├── task5.py
├── training-dataset.csv
├── data_generator.py
└── README.md
```

## **Running the Analysis Tasks**

Open two terminals. The data generator should be running in one terminal while the tasks are running in the other terminal.

### 1) Start the data generator in Terminal 1

```bash
python data_generator.py
# Streams one JSON event per line to localhost:9999
```

### 2) Run the tasks in Terminal 2

Run **one task at a time**:

```bash
# Task 4: Real-Time Fare Prediction Using MLlib Regression
python task4.py

# Task 5: Time-Based Fare Trend Prediction
python task5.py
```

> **Note:** Ensure the training dataset (`training-dataset.csv`) is present in the project directory before running the tasks.

## **Overview**

In this assignment, we extend our real-time analytics pipeline by integrating Machine Learning models using Spark MLlib. We will train regression models offline and apply them to streaming data for real-time predictions and anomaly detection.

## **Objectives**

* **Task 4:** Train a Linear Regression model for fare prediction and apply it to streaming data to detect fare anomalies.
* **Task 5:** Build a time-based fare trend prediction model using windowed aggregations and cyclical time features.

## **Task 4: Real-Time Fare Prediction Using MLlib Regression**

### **Goal**
Train a Linear Regression model to predict fare amounts based on trip distance, then use the model in real-time to detect fare anomalies by comparing predicted vs. actual fares.

### **Approach**

1. **Offline Model Training:** Load training data, use VectorAssembler to prepare distance_km as features, train a LinearRegression model, and save it to models/fare_model.

2. **Real-Time Inference:** Ingest streaming data, load the saved model, apply predictions, and calculate deviation between actual and predicted fares.

### **Sample Output**

```
-------------------------------------------
Batch: 0
-------------------------------------------
+--------------------+-----------+-----------+------------------+------------------+
|             trip_id|distance_km|fare_amount|        prediction|         deviation|
+--------------------+-----------+-----------+------------------+------------------+
|ac6a3544-be6b-4ee...|      29.37|     104.72| 98.76543210987654| 5.953567890123456|
|bd7f4655-cf7c-5ff...|      15.82|      67.45| 65.43210987654321|2.0178901234567893|
+--------------------+-----------+-----------+------------------+------------------+
```

## **Task 5: Time-Based Fare Trend Prediction**

### **Goal**
Predict average fare amounts for future time windows by training a model on historical windowed aggregations with engineered time-based features.

### **Approach**

1. **Offline Model Training:** Load training data, group into 5-minute windows, calculate avg_fare, extract hour_of_day and minute_of_hour features from window.start, train a LinearRegression model, and save it to models/fare_trend_model_v2.

2. **Real-Time Inference:** Apply 5-minute windowed aggregation on streaming data, extract the same time features, load the saved model, and predict avg_fare for each window.

### **Sample Output**

```
-------------------------------------------
Batch: 5
-------------------------------------------
+--------------------+--------------------+------------------+---------------------+
|        window_start|          window_end|          avg_fare|predicted_next_avg...|
+--------------------+--------------------+------------------+---------------------+
|2025-10-14 22:20:...|2025-10-14 22:25:...|  85.6789012345678|    87.12345678901234|
|2025-10-14 22:25:...|2025-10-14 22:30:...|  92.3456789012345|    91.98765432109876|
+--------------------+--------------------+------------------+---------------------+
```

**Obtained Output:**
[Showing output screenshots](./OutputScreenshots)

## **Technical Architecture**

### **Workflow Pipeline:**

1. **Data Generation Layer** - `data_generator.py` simulates real-time ride events via socket (localhost:9999)

2. **Offline Training Layer** - Loads historical data, performs feature engineering and model training, persists models to disk

3. **Stream Processing Layer** - Spark Structured Streaming ingests live data and applies feature transformations

4. **Inference Layer** - Real-time predictions on streaming data with anomaly detection and trend forecasting

5. **Output Layer** - Console output with results visualization

## **Key Technologies Used**

- **Apache Spark Structured Streaming**: Real-time data processing
- **Spark MLlib**: Machine learning model training and inference
- **LinearRegression**: Regression algorithm
- **VectorAssembler**: Feature engineering
- **PySpark**: Python API for Spark

## 📬 Submission Checklist

- [x] Python scripts (`task4.py`, `task5.py`)
- [x] Trained models in the `models/` directory
- [x] Training dataset (`training-dataset.csv`)
- [x] Output screenshots in `OutputScreenshots/` directory
- [x] Completed `README.md` with approach and results
- [x] Commit everything to GitHub repository
- [x] Submit your GitHub repo link on Canvas
