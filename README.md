# AI-Based Government Fund Allocation and Utilization Analysis and Forecasting Using Machine Learning

## 📌 Project Overview

The **AI-Based Government Fund Allocation and Utilization Analysis and Forecasting** project is a machine learning-based system designed to analyze historical government fund allocation, evaluate fund utilization efficiency, and forecast future funding trends.

The project integrates traditional Machine Learning algorithms with advanced optimization techniques such as:

- Machine Learning (ML)
- Artificial Immune System (AIS)
- Particle Swarm Optimization (PSO)

The system analyzes historical financial data, identifies allocation and utilization patterns, evaluates model performance, and generates experimental predictions for future government funding.

It also provides graphical visualizations, correlation analysis, model comparison, and downloadable prediction results.

> **Note:** The current dataset contains only four complete annual observations. Therefore, the reported model scores represent training performance rather than independently validated prediction accuracy. Future forecasts are experimental.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. Analyze historical government fund allocation.
2. Study year-wise fund utilization trends.
3. Identify underutilization and overutilization.
4. Calculate government fund utilization efficiency.
5. Implement machine learning regression models.
6. Apply Artificial Immune System optimization.
7. Apply Particle Swarm Optimization.
8. Compare different model performances.
9. Generate experimental future funding predictions.
10. Visualize allocation and utilization trends.
11. Save trained models for future use.
12. Generate CSV reports and graphical results.

---

## 📊 Dataset Description

The project uses the dataset:

`RS_Session_270_AU_2397_B.ii_.csv`

### Dataset Features

| Feature | Description |
|---------|-------------|
| Year | Financial year |
| Funds Allocated | Government funds allocated |
| Funds Utilised | Government funds utilized |

### Dataset Characteristics

- Data Type: Structured CSV
- Domain: Government Finance
- Learning Type: Supervised Regression
- Input Feature: Financial Year
- Target Variables: Funds Allocated and Funds Utilised
- Complete Historical Records: 4
- Missing Values: Present
- Forecasting Type: Annual Time-Series Estimation

### Sample Dataset

| Financial Year | Funds Allocated | Funds Utilised |
|----------------|-----------------|----------------|
| 2020-21 | 129 | 154 |
| 2021-22 | 129 | 116 |
| 2022-23 | 148 | 146 |
| 2023-24 | 113 | 111 |
| 2024-25 | Missing | Missing |

The dataset is preprocessed before model training.

Missing target values are excluded from model training and retained as missing observations.

---

## 🛠️ Technologies Used

### Programming Language

- Python 3.10+

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Joblib
- PyYAML
- h5py

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Python
- Git
- GitHub

---

## 🤖 Machine Learning Algorithms

The project implements the following algorithms.

### 1. Linear Regression

Linear Regression models the relationship between financial year and government funding.

It estimates funding trends using a linear equation.

**Equation:**

Y = β₀ + β₁X

Where:

- Y = Predicted fund allocation or utilization
- X = Financial year
- β₀ = Intercept
- β₁ = Regression coefficient

### 2. Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It is used to analyze nonlinear relationships in historical government funding data.

### 3. Decision Tree Regressor

Decision Tree Regression predicts funding values by recursively dividing the dataset into decision regions.

### 4. Gradient Boosting Regressor

Gradient Boosting combines multiple weak learners to minimize prediction errors.

### 5. Support Vector Regression

Support Vector Regression estimates funding values using support vector-based regression techniques.

### 6. Artificial Neural Network

A feedforward neural network is implemented using TensorFlow.

Architecture:

Input Layer → Dense(32) → Dense(16) → Output(2)

The output layer predicts:

- Funds Allocated
- Funds Utilised

---

# 🧬 Artificial Immune System (AIS)

## Overview

Artificial Immune System is a nature-inspired optimization technique based on the behavior of biological immune systems.

The project implements the **Clonal Selection Algorithm (CLONALG)** to optimize polynomial regression coefficients.

### AIS Workflow

1. Initialize antibody population.
2. Calculate fitness values.
3. Select high-quality antibodies.
4. Generate antibody clones.
5. Apply mutation.
6. Replace weak antibodies.
7. Repeat optimization.
8. Select the best antibody.
9. Generate funding predictions.

### AIS Fitness Function

Fitness = MSE + λΣ(w²)

Where:

- MSE = Mean Squared Error
- λ = Regularization coefficient
- w = Model coefficients

The algorithm attempts to minimize prediction error.

### AIS Configuration

| Parameter | Value |
|-----------|-------|
| Population Size | 50 |
| Generations | 200 |
| Selection Size | 10 |
| Clone Factor | 5 |
| Mutation Rate | 0.15 |
| Random Replacement | 5 |
| Polynomial Degree | 2 |

### AIS Generated Outputs

- `ais_accuracy_graph.png`
- `ais_comparison_graph.png`
- `ais_heatmap.png`
- `ais_result.csv`
- `ais_result_graph.png`
- `ais_prediction.csv`
- `ais_prediction_graph.png`
- `ais_convergence_graph.png`
- `ais_utilization_graph.png`
- `ais_utilization_analysis.csv`
- `ais_optimization_history.csv`
- `ais_model.pkl`
- `ais_model.h5`
- `ais_config.yaml`
- `ais_results.json`

---

# 🐦 Particle Swarm Optimization (PSO)

## Overview

Particle Swarm Optimization is a population-based optimization algorithm inspired by the collective movement of birds and fish.

PSO optimizes regression coefficients by updating particle positions and velocities.

Each particle represents a candidate solution.

### PSO Workflow

1. Initialize particle population.
2. Initialize particle velocities.
3. Calculate particle fitness.
4. Update personal best positions.
5. Update global best position.
6. Calculate new particle velocities.
7. Update particle positions.
8. Repeat optimization.
9. Select the best solution.
10. Generate funding predictions.

### PSO Velocity Equation

v(t+1) = w × v(t) + c₁r₁(pbest − x) + c₂r₂(gbest − x)

### PSO Position Equation

x(t+1) = x(t) + v(t+1)

Where:

- w = Inertia weight
- c₁ = Cognitive coefficient
- c₂ = Social coefficient
- pbest = Personal best position
- gbest = Global best position
- x = Particle position
- v = Particle velocity

### PSO Configuration

| Parameter | Value |
|-----------|-------|
| Particles | 50 |
| Iterations | 200 |
| Inertia Weight | 0.7 |
| Cognitive Coefficient | 1.5 |
| Social Coefficient | 1.5 |
| Polynomial Degree | 2 |
| Regularization | 0.001 |

### PSO Generated Outputs

- `pso_accuracy_graph.png`
- `pso_comparison_graph.png`
- `pso_heatmap.png`
- `pso_result.csv`
- `pso_result_graph.png`
- `pso_prediction.csv`
- `pso_prediction_graph.png`
- `pso_convergence_graph.png`
- `pso_scatter_graph.png`
- `pso_utilization_graph.png`
- `pso_utilization_analysis.csv`
- `pso_optimization_history.csv`
- `pso_model_comparison.csv`
- `pso_model.pkl`
- `pso_model.h5`
- `pso_config.yaml`
- `pso_results.json`

---

# ⚙️ Project Workflow

The complete project follows the workflow:

Dataset Collection
        |
        v
Data Preprocessing
        |
        v
Feature Extraction
        |
        v
Feature Scaling
        |
        v
Machine Learning Model Training
        |
        v
AIS Optimization
        |
        v
PSO Optimization
        |
        v
Model Evaluation
        |
        v
Historical Results
        |
        v
Future Forecasting
        |
        v
Data Visualization
        |
        v
CSV and Model Export

---

# 📈 Model Evaluation Metrics

The project evaluates regression models using the following metrics.

## 1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

MAE = (1/n) × Σ|Actual − Predicted|

Lower MAE indicates a better fit.

## 2. Mean Squared Error (MSE)

MSE calculates the average squared prediction error.

MSE = (1/n) × Σ(Actual − Predicted)²

Lower MSE indicates a better fit.

## 3. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

RMSE = √MSE

Lower RMSE indicates a better fit.

## 4. R² Score

R² measures how well the regression model explains variation in the observed data.

R² = 1 − (SS_res / SS_tot)

A higher R² generally indicates a better fit to the evaluated observations.

R² is not classification accuracy and can be negative.

### Evaluation Limitation

The current implementation evaluates predictions on the same records used for training.

Therefore, the generated metrics are **in-sample training-fit scores**, not independent test results.

A larger dataset and chronological validation are required to establish forecasting reliability.

---

# 📊 Results and Visualizations

The project generates multiple graphs to analyze government fund allocation and utilization.

## 1. Model Accuracy Graph

The following graph displays the R² training-fit scores of different machine learning models.

![Machine Learning Model Accuracy](accuracy_graph.png)

**Description:**

- Compares regression model training-fit scores.
- Helps identify models that fit the historical observations more closely.
- Higher R² indicates better in-sample fit.
- Does not establish real-world prediction accuracy.

---

## 2. Machine Learning Model Comparison

![Model Comparison](comparison_graph.png)

The comparison graph evaluates models using:

- Mean Absolute Error
- Root Mean Squared Error

Lower error values indicate better fit to the historical data.

---

## 3. Correlation Heatmap

![Correlation Heatmap](heatmap.png)

The heatmap visualizes correlations among:

- Financial Year
- Funds Allocated
- Funds Utilised

It helps explore relationships between numerical variables.

Because the dataset contains very few observations, correlations may be unstable.

---

## 4. Historical Result Visualization

![Historical Results](result_graph.png)

This graph compares actual and model-predicted government funding values.

It displays:

- Actual Funds Allocated
- Actual Funds Utilised
- Predicted Funds Allocated
- Predicted Funds Utilised

The corresponding values are stored in:

`result.csv`

---

## 5. Future Prediction Visualization

![Future Predictions](prediction_graph.png)

This graph displays historical government funding data alongside experimental future predictions.

The predictions are generated using trained regression models.

The corresponding values are stored in:

`prediction.csv`

---

## 6. Neural Network Training Loss

![Training Loss](training_loss_graph.png)

This graph shows how the neural network training loss changes across epochs.

It helps visualize the optimization process.

---

## 7. Government Fund Utilization Analysis

![Fund Utilization](fund_utilization_graph.png)

This graph displays year-wise government fund utilization percentages.

### Fund Utilization Formula

Utilization Rate (%) = (Funds Utilised / Funds Allocated) × 100

A utilization rate below 100% indicates that utilized funds were lower than allocated funds.

A utilization rate above 100% indicates that reported utilization exceeded the reported allocation.

Such cases should be investigated before drawing financial conclusions.

---

# 🧬 AIS Results and Visualizations

## AIS Training-Fit Graph

![AIS Accuracy](ais_accuracy_graph.png)

Displays the AIS regression model's R² training-fit scores.

## AIS Error Comparison

![AIS Comparison](ais_comparison_graph.png)

Displays AIS model error metrics.

## AIS Correlation Heatmap

![AIS Heatmap](ais_heatmap.png)

Displays correlations among historical funding variables.

## AIS Historical Results

![AIS Historical Results](ais_result_graph.png)

Compares historical funding values with AIS-fitted values.

## AIS Future Predictions

![AIS Predictions](ais_prediction_graph.png)

Displays experimental funding forecasts generated by the AIS-optimized regression model.

## AIS Convergence

![AIS Convergence](ais_convergence_graph.png)

Shows changes in optimization fitness across generations.

The graph helps analyze whether the clonal selection process converges.

---

# 🐦 PSO Results and Visualizations

## PSO Training-Fit Graph

![PSO Accuracy](pso_accuracy_graph.png)

Displays the PSO regression model's R² training-fit scores.

## PSO Error Comparison

![PSO Comparison](pso_comparison_graph.png)

Displays MAE and RMSE for the PSO model.

## PSO Correlation Heatmap

![PSO Heatmap](pso_heatmap.png)

Displays relationships between historical funding variables.

## PSO Historical Results

![PSO Historical Results](pso_result_graph.png)

Compares actual historical funding values with PSO-fitted predictions.

## PSO Future Predictions

![PSO Predictions](pso_prediction_graph.png)

Displays experimental government funding forecasts generated using PSO.

## PSO Convergence

![PSO Convergence](pso_convergence_graph.png)

Displays the optimization fitness across PSO iterations.

## PSO Actual vs Predicted Scatter Plot

![PSO Scatter](pso_scatter_graph.png)

Shows the relationship between actual and predicted funding values.

---

# 📁 Project Directory Structure

```text
AI-Based Government Fund Allocation and Utilization Analysis and Forecasting/
│
├── RS_Session_270_AU_2397_B.ii_.csv
│
├── government_fund_analysis.py
├── ais_government_fund_analysis.py
├── pso_government_fund_analysis.py
│
├── README.md
│
├── Machine Learning Models
│   ├── government_fund_model.h5
│   ├── government_fund_model.pkl
│   ├── scalers.pkl
│   ├── model_config.yaml
│   └── model_results.json
│
├── Machine Learning Results
│   ├── accuracy_graph.png
│   ├── comparison_graph.png
│   ├── heatmap.png
│   ├── result.csv
│   ├── result_graph.png
│   ├── prediction.csv
│   ├── prediction_graph.png
│   ├── model_comparison.csv
│   ├── training_loss_graph.png
│   ├── training_history.csv
│   ├── fund_utilization_analysis.csv
│   └── fund_utilization_graph.png
│
├── AIS Results
│   ├── ais_model.pkl
│   ├── ais_model.h5
│   ├── ais_scalers.pkl
│   ├── ais_config.yaml
│   ├── ais_results.json
│   ├── ais_accuracy_graph.png
│   ├── ais_comparison_graph.png
│   ├── ais_heatmap.png
│   ├── ais_result.csv
│   ├── ais_result_graph.png
│   ├── ais_prediction.csv
│   ├── ais_prediction_graph.png
│   ├── ais_convergence_graph.png
│   ├── ais_utilization_analysis.csv
│   ├── ais_utilization_graph.png
│   └── ais_optimization_history.csv
│
└── PSO Results
    ├── pso_model.pkl
    ├── pso_model.h5
    ├── pso_scalers.pkl
    ├── pso_config.yaml
    ├── pso_results.json
    ├── pso_accuracy_graph.png
    ├── pso_comparison_graph.png
    ├── pso_heatmap.png
    ├── pso_result.csv
    ├── pso_result_graph.png
    ├── pso_prediction.csv
    ├── pso_prediction_graph.png
    ├── pso_convergence_graph.png
    ├── pso_scatter_graph.png
    ├── pso_utilization_analysis.csv
    ├── pso_utilization_graph.png
    ├── pso_optimization_history.csv
    └── pso_model_comparison.csv
```

**Note:** The directory groups above are logical categories. The supplied Python scripts save their output files directly in the main project directory.

---

# 🚀 Installation and Setup

## Step 1: Install Python

Install Python 3.10 or a compatible version.

Verify the installation:

```bash
python --version
```

## Step 2: Open the Project Directory

```bat
cd "C:\Users\sagni\Downloads\AI-Based Government Fund Allocation and Utilization Analysis and Forecasting"
```

## Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bat
venv\Scripts\activate
```

## Step 4: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib pyyaml h5py
```

## Step 5: Run the Machine Learning Model

```bash
python government_fund_analysis.py
```

This script trains the machine learning models and generates the corresponding results.

## Step 6: Run the AIS Model

```bash
python ais_government_fund_analysis.py
```

This script executes the Artificial Immune System optimization algorithm.

All generated AIS outputs use the `ais_` prefix.

## Step 7: Run the PSO Model

```bash
python pso_government_fund_analysis.py
```

This script executes Particle Swarm Optimization.

All generated PSO outputs use the `pso_` prefix.

---

# 💾 Model Storage

The project saves trained models and configuration information in different formats.

| Format | Purpose |
|--------|---------|
| `.pkl` | Stores trained models and preprocessing objects |
| `.h5` | Stores neural network models or optimized model weights |
| `.yaml` | Stores model configurations |
| `.json` | Stores evaluation results and predictions |
| `.csv` | Stores historical results and forecasts |
| `.png` | Stores generated visualizations |

### Important Model Format Information

The conventional machine learning implementation saves a TensorFlow neural network in `.h5` format.

The AIS and PSO implementations save optimized regression coefficients in HDF5 format.

AIS and PSO `.h5` files are not Keras models and must be loaded using `h5py`.

---

# 📉 Government Fund Utilization Analysis

Fund utilization efficiency is calculated using:

Utilization Percentage = (Funds Utilised / Funds Allocated) × 100

### Example

Suppose:

Funds Allocated = 148

Funds Utilised = 146

Then:

Utilization Percentage = (146 / 148) × 100

Utilization Percentage = 98.65%

This indicates that approximately 98.65% of the reported allocated funds were utilized.

The system also calculates the funding gap.

### Funding Gap Formula

Funding Gap = Funds Allocated − Funds Utilised

A positive funding gap indicates that allocated funds exceed utilized funds.

A negative funding gap indicates that reported utilization exceeds allocation.

---

# 🔮 Future Forecasting

The project generates experimental predictions for future financial years.

The forecasting models use historical financial year information to estimate:

1. Future Funds Allocated
2. Future Funds Utilised
3. Funding Gaps
4. Fund Utilization Percentages

### Forecasting Output

The generated prediction files include:

- `prediction.csv`
- `ais_prediction.csv`
- `pso_prediction.csv`

These files can be used for further visualization and exploratory analysis.

### Forecasting Limitations

The dataset contains too few complete observations to establish reliable future trends.

Quadratic regression may produce unrealistic extrapolations, including negative funding values.

Tree-based regression models may produce nearly constant forecasts beyond the observed years.

Forecasts should therefore be interpreted as demonstrations of model behavior rather than reliable financial estimates.

---

# 📊 ML vs AIS vs PSO Comparison

The project supports comparisons between conventional machine learning, AIS, and PSO.

| Feature | Machine Learning | AIS | PSO |
|---------|------------------|-----|-----|
| Regression Modeling | Yes | Yes | Yes |
| Optimization | Model-dependent | Clonal Selection | Swarm Optimization |
| Historical Analysis | Yes | Yes | Yes |
| Future Prediction | Yes | Yes | Yes |
| R² Evaluation | Yes | Yes | Yes |
| MAE Evaluation | Yes | Yes | Yes |
| RMSE Evaluation | Yes | Yes | Yes |
| Convergence Analysis | Neural Network Loss | AIS Fitness | PSO Fitness |
| Model Export | Yes | Yes | Yes |
| Visualization | Yes | Yes | Yes |

### Comparison Criteria

The algorithms can be compared using:

- MAE
- MSE
- RMSE
- R² Score
- Optimization Convergence
- Historical Prediction Error
- Forecast Stability

### Important Consideration

Training-fit metrics alone cannot establish which algorithm performs best on unseen data.

A reliable comparison requires more observations, consistent training conditions, and chronological out-of-sample evaluation.

---

# 🌟 Key Features

- Automated CSV data loading
- Data preprocessing
- Missing-value identification
- Financial year extraction
- Historical funding analysis
- Multiple machine learning models
- Artificial Immune System optimization
- Particle Swarm Optimization
- Regression performance evaluation
- Correlation heatmap visualization
- Actual vs predicted graphs
- Future funding estimation
- Fund utilization analysis
- Optimization convergence graphs
- CSV result generation
- Model serialization
- YAML configuration export
- JSON result export
- High-resolution graphical output

---

# 🔬 Applications

The proposed system can support exploratory research in:

### 1. Government Financial Analysis

Analyzing historical government fund allocation and utilization patterns.

### 2. Public Sector Research

Studying financial allocation trends across reporting periods.

### 3. Budget Utilization Monitoring

Identifying potential differences between allocated and utilized funds.

### 4. Financial Data Visualization

Presenting government financial information using graphs and charts.

### 5. Machine Learning Research

Comparing conventional regression models with nature-inspired optimization algorithms.

### 6. Educational Forecasting Demonstrations

Demonstrating regression, optimization, model evaluation, and time-series extrapolation.

---

# ⚠️ Limitations

The project has the following limitations:

1. The dataset contains only four complete historical observations.
2. Model scores are calculated using training data.
3. Independent test accuracy is not available.
4. Future predictions are experimental.
5. Financial year is the only input feature.
6. The dataset does not contain additional economic indicators.
7. Quadratic models may extrapolate poorly.
8. Model comparisons do not establish real-world forecasting superiority.
9. Correlation analysis is statistically unstable with very small datasets.
10. Forecasting reliability requires additional historical information.

---

# 🚀 Future Enhancements

The project can be improved through:

1. Collection of additional historical financial records.
2. Integration of state-wise and district-wise funding datasets.
3. Inclusion of government scheme categories.
4. Addition of inflation and economic indicators.
5. Chronological train/test validation.
6. Walk-forward forecasting evaluation.
7. Hyperparameter optimization.
8. Time-series forecasting using ARIMA.
9. LSTM-based forecasting with sufficient data.
10. Interactive dashboards using Streamlit.
11. Automated reporting systems.
12. Explainable AI techniques.
13. Model uncertainty estimation.
14. Financial anomaly detection.
15. Integration with regularly updated public financial datasets.

---

# 📌 Expected Outcomes

The project is designed to produce:

- Historical fund allocation analysis
- Fund utilization analysis
- Machine learning regression models
- AIS-optimized regression models
- PSO-optimized regression models
- Model training-fit comparisons
- Historical prediction visualizations
- Experimental future funding forecasts
- Correlation heatmaps
- Optimization convergence graphs
- CSV reports
- Saved model artifacts
- Configuration and result files

---

# 🏁 Conclusion

The **AI-Based Government Fund Allocation and Utilization Analysis and Forecasting Using Machine Learning** project demonstrates how data analytics, machine learning, and nature-inspired optimization algorithms can be applied to government financial datasets.

The project integrates conventional regression algorithms with Artificial Immune System and Particle Swarm Optimization techniques.

It provides tools for analyzing historical government funding patterns, evaluating utilization efficiency, visualizing financial trends, and generating experimental future predictions.

The implementation also demonstrates model optimization, regression evaluation, and automated result generation.

Although the current dataset is too small for reliable forecasting, the project provides a foundation for future research using larger and more comprehensive government financial datasets.

---

## 📜 License

This project is intended for educational, academic, and research purposes.

## ⭐ Project Highlights

**Machine Learning | Artificial Intelligence | Government Finance | Fund Allocation | Fund Utilization | Financial Forecasting | Artificial Immune System | Particle Swarm Optimization | Regression Analysis | Data Visualization | Python**
