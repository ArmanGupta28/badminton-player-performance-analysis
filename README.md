# 🏸 Badminton Player Performance Analysis

An interactive data science project for analyzing professional badminton match results using Python, statistical methods, data visualization, and machine learning.

## 📌 Project Overview

**Badminton Player Performance Analysis** explores professional badminton match data to understand match outcomes, scoring patterns, player performance, and relationships between match variables.

The project combines exploratory data analysis, statistical hypothesis testing, regression, classification, and an interactive Streamlit dashboard to present insights in an accessible format.

## 🎯 Objectives

- Analyze professional badminton match results.
- Study scoring patterns and total points played per match.
- Identify players with the highest number of recorded wins.
- Visualize match statistics using charts and graphs.
- Apply probability distributions, sampling, and the Central Limit Theorem (CLT).
- Perform confidence interval estimation and statistical hypothesis testing.
- Analyze relationships between numerical variables using correlation and covariance.
- Apply chi-square tests and two-way ANOVA.
- Build regression and classification models for machine learning practice.
- Develop an interactive dashboard using Streamlit.

## 📊 Dataset

The project uses professional badminton match data from the following public GitHub repository:

**Dataset source:** [BWF Match Data — SahilMotyar](https://github.com/SahilMotyar/bwf-match-data)

The dataset contains match-level information, including fields such as:

- `date` — match date
- `tournament` — tournament name
- `team1` — first player or team
- `team2` — second player or team
- `winner` — recorded winner position
- `score` — match score
- `discipline` — match category, such as men's singles

The analysis focuses on **men's singles matches**.

### Data preprocessing

The dataset is prepared using the following steps:

1. Load the match data using Pandas.
2. Convert date values into a suitable datetime format.
3. Filter men's singles matches.
4. Remove duplicate records.
5. Handle missing values in essential columns.
6. Retain valid winner labels.
7. Extract game scores and calculate points for both players.
8. Calculate total match points and point difference.
9. Count the number of games played in each match.
10. Prepare the cleaned dataset for statistical analysis and visualization.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and heatmaps |
| SciPy | Statistical tests and probability calculations |
| Statsmodels | ANOVA and statistical modeling |
| Scikit-learn | Regression and classification |
| Streamlit | Interactive dashboard |
| GitHub | Source code and project version control |

## 🔬 Methodology

The project follows a structured data science workflow:

1. **Data collection:** Obtain professional badminton match data from a public dataset.
2. **Data preprocessing:** Clean, filter, and transform the match records.
3. **Feature engineering:** Calculate scoring variables, total points, point differences, and games played.
4. **Exploratory data analysis:** Calculate descriptive statistics and visualize important patterns.
5. **Statistical analysis:** Apply sampling, CLT, confidence intervals, hypothesis tests, and relationship analysis.
6. **Machine learning:** Experiment with regression and classification algorithms.
7. **Dashboard development:** Present selected results through an interactive Streamlit application.
8. **Interpretation:** Examine results while considering data limitations and model assumptions.

## 📈 Statistical Analysis

The project covers the following statistical techniques:

### 1. Descriptive Statistics
- Mean, median, and mode
- Variance and standard deviation
- Quartiles and interquartile range
- Minimum, maximum, and summary statistics

### 2. Data Visualization
- Histograms
- Boxplots
- Bar charts
- Pie charts
- Scatter plots
- Line charts
- Correlation heatmaps

### 3. Probability Distributions and Sampling
- Explore the distribution of total match points.
- Select random samples from the match dataset.
- Examine sampling distributions of sample means.
- Demonstrate the Central Limit Theorem (CLT).

### 4. Confidence Intervals and Hypothesis Testing
- Estimate a population mean using a 95% confidence interval.
- Perform a one-sample t-test.
- Demonstrate a large-sample Z-test using an estimated standard deviation.

### 5. Chi-Square Test
Study the association between winner position and the number of games played using a contingency table.

### 6. Correlation and Covariance
- Calculate covariance and correlation matrices.
- Apply Pearson correlation.
- Apply Spearman rank correlation.
- Visualize relationships between scoring variables.

### 7. Two-Way ANOVA
Examine the relationship between total match points, games played, and winner position, including their interaction.

**Note:** Statistical results must be interpreted with appropriate assumptions. For example, scoring variables are related by construction, and match observations may not be independent.

## 🤖 Machine Learning

The project demonstrates regression and classification using Scikit-learn.

### Regression

The regression experiments include:

- Linear Regression
- Ridge Regression
- Lasso Regression

The models explore relationships between scoring variables and total match characteristics.

### Classification

The classification experiments include:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)

The classification workflow uses historical player win rates as features to demonstrate match-outcome prediction.

### Model Evaluation

Models are evaluated using suitable metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Regression models can also be evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R², where implemented.

**Important:** These experiments demonstrate analytical and machine-learning techniques. Match scores and results are recorded after a match, so they should not be treated as pre-match prediction features. Historical win rates provide a more suitable starting point, but the evaluation remains subject to temporal and data-dependency limitations.

## 🖥️ Interactive Streamlit Dashboard

The Streamlit dashboard provides a user-friendly interface for exploring the badminton dataset.

Depending on the implemented features, the dashboard can include:

- Dataset overview and summary metrics
- Date or year filters
- Match data preview
- Top players by recorded wins
- Player performance comparisons
- Match scoring distributions
- Correlation heatmaps
- Interactive charts and tables

### Live Dashboard

**Streamlit App:** [Add your deployed Streamlit URL here]

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/badminton-player-performance-analysis.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the project folder

```bash
cd badminton-player-performance-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## 📁 Project Structure

```text
badminton-player-performance-analysis/
│
├── app.py
├── badminton_cleaned.csv
├── requirements.txt
└── README.md
```

If a separate Jupyter notebook is included in the repository, add its actual filename to this structure.

## 📋 Results and Expected Outcomes

The project is designed to produce insights into:

- The distribution of points scored during matches.
- Differences in scoring between match participants.
- Players with the highest recorded match-win counts.
- Relationships among match scoring variables.
- Statistical evidence from hypothesis tests.
- Comparative performance metrics for experimental machine-learning models.
- Interactive exploration of match data through the dashboard.

Actual numerical findings should be reported after examining the executed notebook and model outputs.

## ⚠️ Limitations

- The dataset primarily contains match-level results and scores.
- It does not necessarily contain detailed shot-level statistics, smash accuracy, unforced errors, or net-play success.
- Recorded wins do not alone measure a player's overall skill.
- Statistical tests depend on their assumptions and the structure of the data.
- Repeated appearances by the same players can introduce dependencies between observations.
- Machine-learning performance may be affected by temporal ordering, feature availability, and dataset imbalance.

## 🚀 Future Scope

Potential improvements include:

- Add more interactive dashboard filters.
- Include additional badminton disciplines, such as women's singles and doubles.
- Incorporate player rankings and tournament-level information where available.
- Develop a more rigorous time-based evaluation for match-outcome prediction.
- Add model comparison charts and feature-importance analysis.
- Integrate richer shot-level or rally-level data if a reliable dataset becomes available.
- Deploy the application publicly and improve its user interface.

## 👥 Contributors

Add the names of the project team members here.

## 📄 License and Attribution

The original dataset is maintained in the [BWF Match Data repository](https://github.com/SahilMotyar/bwf-match-data). Refer to its license and attribution requirements before redistributing the dataset.

This project is intended for educational and analytical purposes.

---

**Built with Python, Pandas, Scikit-learn, and Streamlit.**
