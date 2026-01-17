# Bengaluru Traffic — Data Science in Action

Traffic in Bengaluru isn’t random — it follows clear, measurable patterns.
This project uses data science and machine learning to understand where congestion happens, why it happens, and what can be done about it.

## What This Project Does

- Analyzes traffic patterns across major Bengaluru areas and roads
- Identifies congestion hotspots using data
- Visualizes relationships between traffic volume, speed, capacity, and delays
- Predicts congestion levels using a machine learning model
- Detects unusual traffic events like accidents or sudden jams

## What’s Inside

Complete notebook with:

- Data cleaning & feature engineering
- Exploratory Data Analysis (EDA)
- Visualizations (saved as PNGs)
- Machine learning model
- Anomaly detection

Dataset
Realistic traffic data including volume, speed, congestion, incidents, weather, and road conditions.

## Machine Learning Model (Why It Matters)

A Random Forest Classifier is used to predict congestion levels
(Low / Medium / High) based on factors like:

- Traffic volume
- Average speed
- Road capacity utilization
- Parking usage
- Public transport usage
- Incident reports

### Why this model?

- Works well with real-world, noisy data
- Captures non-linear traffic behavior
- Shows which factors contribute most to congestion
Accuracy: ~94%

## Real-world use:

- Predict congestion before it worsens
- Adjust traffic signals dynamically
- Alert authorities about high-risk roads
- Support better urban planning decisions

## Extra Insight: Anomaly Detection

Using Isolation Forest, the project also flags:

- Sudden traffic spikes
- Accident-prone days
- Unusual slowdowns

This helps in early incident detection, not just prediction.

## Key Takeaway

Bengaluru’s traffic problem is not a mystery — it’s a data problem.

With simple analysis, visualizations, and ML:

- We can spot congestion patterns
- Understand their causes
- And make smarter, data-backed decisions

## Tools Used

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-learn · Jupyter Notebook
