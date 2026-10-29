
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, ttest_1samp, pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Badminton Performance Analysis",
    page_icon="🏸",
    layout="wide"
)

st.title("🏸 Badminton Player Performance Analysis")
st.caption(
    "Professional badminton match analysis using statistics, "
    "visualization and machine learning."
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    data = pd.read_csv("badminton_cleaned.csv")

    required = ["team1", "team2", "winner"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    data["winner"] = pd.to_numeric(data["winner"], errors="coerce")

    if "total_match_points" not in data.columns:
        if "score" in data.columns:
            def get_total(score):
                if pd.isna(score):
                    return np.nan
                pairs = re.findall(r"(\d+)-(\d+)", str(score))
                if not pairs:
                    return np.nan
                return sum(int(a) + int(b) for a, b in pairs)

            data["total_match_points"] = data["score"].apply(get_total)
        else:
            data["total_match_points"] = np.nan

    if "team1_points" not in data.columns and "score" in data.columns:
        def get_points(score):
            if pd.isna(score):
                return pd.Series([np.nan, np.nan])
            pairs = re.findall(r"(\d+)-(\d+)", str(score))
            if not pairs:
                return pd.Series([np.nan, np.nan])
            p1 = sum(int(a) for a, b in pairs)
            p2 = sum(int(b) for a, b in pairs)
            return pd.Series([p1, p2])

        data[["team1_points", "team2_points"]] = (
            data["score"].apply(get_points)
        )

    if "games_played" not in data.columns and "score" in data.columns:
        data["games_played"] = data["score"].apply(
            lambda s: len(re.findall(r"\d+-\d+", str(s)))
            if pd.notna(s) else np.nan
        )

    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info(
        "Ensure badminton_cleaned.csv is uploaded to the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

# Basic validity filters for match-level analysis
df = df[
    df["team1"].notna()
    & df["team2"].notna()
    & df["winner"].isin([1, 2])
].copy()

if df.empty:
    st.error("No valid matches were found in the dataset.")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Dashboard Filters")

filtered = df.copy()

if "date" in filtered.columns and filtered["date"].notna().any():
    years = sorted(filtered["date"].dropna().dt.year.unique())

    if years:
        selected_years = st.sidebar.multiselect(
            "Select year(s)",
            years,
            default=years
        )

        if selected_years:
            filtered = filtered[
                filtered["date"].dt.year.isin(selected_years)
            ]
        else:
            filtered = filtered.iloc[0:0]

if "tournament" in filtered.columns:
    tournaments = sorted(
        filtered["tournament"].dropna().astype(str).unique()
    )

    if tournaments:
        selected_tournaments = st.sidebar.multiselect(
            "Tournament",
            tournaments,
            default=[]
        )
        if selected_tournaments:
            filtered = filtered[
                filtered["tournament"].astype(str).isin(
                    selected_tournaments
                )
            ]

if filtered.empty:
    st.warning("No matches match the selected filters.")
    st.stop()

# ---------------- HELPER FUNCTIONS ----------------

def player_wins(data):
    wins1 = data.loc[data["winner"] == 1, "team1"]
    wins2 = data.loc[data["winner"] == 2, "team2"]
    return pd.concat([wins1, wins2]).value_counts()

def player_matches(data):
    appearances = pd.concat([data["team1"], data["team2"]])
    return appearances.value_counts()

def get_player_summary(data):
    wins = player_wins(data)
    appearances = player_matches(data)

    summary = pd.DataFrame({
        "Matches": appearances,
        "Wins": wins
    }).fillna(0)

    summary["Losses"] = summary["Matches"] - summary["Wins"]
    summary["Win Rate (%)"] = (
        summary["Wins"] / summary["Matches"] * 100
    )
    return summary.sort_values("Wins", ascending=False)

# ---------------- TABS ----------------

overview, players, match_analysis, statistics, machine_learning = st.tabs([
    "📊 Overview",
    "🏆 Player Analysis",
    "🎯 Match Analysis",
    "📐 Statistical Analysis",
    "🤖 Machine Learning"
])

# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview:
    st.header("Dataset Overview")

    total_matches = len(filtered)
    total_players = pd.concat([
        filtered["team1"], filtered["team2"]
    ]).nunique()

    total_tournaments = (
        filtered["tournament"].nunique()
        if "tournament" in filtered.columns else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches", f"{total_matches:,}")
    c2.metric("Unique Players", f"{total_players:,}")
    c3.metric("Tournaments", f"{total_tournaments:,}")

    st.subheader("Dataset Preview")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.subheader("Numerical Summary")
    numeric = filtered.select_dtypes(include=np.number)

    if not numeric.empty:
        st.dataframe(numeric.describe().T, use_container_width=True)

    st.subheader("Missing Values")
    missing = filtered.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing):
        st.dataframe(
            missing.rename("Missing Count").to_frame(),
            use_container_width=True
        )
    else:
        st.success("No missing values in the currently filtered data.")

# ==================================================
# TAB 2: PLAYER ANALYSIS
# ==================================================

with players:
    st.header("Player Performance")

    summary = get_player_summary(filtered)

    st.subheader("Top 10 Players by Recorded Wins")
    top10 = summary.head(10).sort_values("Wins")

    if not top10.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top10.index.astype(str), top10["Wins"])
        ax.set_xlabel("Recorded Wins")
        ax.set_ylabel("Player")
        ax.set_title("Top 10 Players by Match Wins")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Player Statistics")
    st.dataframe(
        summary.reset_index(names="Player"),
        use_container_width=True
    )

    st.subheader("Compare Two Players")
    player_names = sorted(summary.index.astype(str).unique())

    if len(player_names) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            p1 = st.selectbox("First player", player_names, index=0)

        with col2:
            p2 = st.selectbox(
                "Second player",
                player_names,
                index=1
            )

        if p1 == p2:
            st.warning("Select two different players.")
        else:
            for p in [p1, p2]:
                row = summary.loc[p]
                st.write(
                    f"**{p}** — Matches: {int(row['Matches'])}, "
                    f"Wins: {int(row['Wins'])}, "
                    f"Win rate: {row['Win Rate (%)']:.2f}%"
                )

            head_to_head = filtered[
                (
                    (filtered["team1"].astype(str) == p1)
                    & (filtered["team2"].astype(str) == p2)
                )
                | (
                    (filtered["team1"].astype(str) == p2)
                    & (filtered["team2"].astype(str) == p1)
                )
            ]

            p1_wins = (
                ((head_to_head["team1"].astype(str) == p1)
                 & (head_to_head["winner"] == 1))
                | ((head_to_head["team2"].astype(str) == p1)
                   & (head_to_head["winner"] == 2))
            ).sum()

            p2_wins = (
                ((head_to_head["team1"].astype(str) == p2)
                 & (head_to_head["winner"] == 1))
                | ((head_to_head["team2"].astype(str) == p2)
                   & (head_to_head["winner"] == 2))
            ).sum()

            st.metric("Head-to-head matches", len(head_to_head))
            st.write(f"**{p1} wins in these matches:** {p1_wins}")
            st.write(f"**{p2} wins in these matches:** {p2_wins}")

            if len(head_to_head) == 0:
                st.info("No head-to-head matches found in the filtered data.")

# ==================================================
# TAB 3: MATCH ANALYSIS
# ==================================================

with match_analysis:
    st.header("Match Scoring Analysis")

    if "total_match_points" in filtered.columns:
        points = pd.to_numeric(
            filtered["total_match_points"], errors="coerce"
        ).dropna()

        if not points.empty:
            a, b, c = st.columns(3)
            a.metric("Average Match Points", f"{points.mean():.2f}")
            b.metric("Median Match Points", f"{points.median():.2f}")
            c.metric("Standard Deviation", f"{points.std():.2f}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribution of Total Points")
                fig, ax = plt.subplots()
                ax.hist(points, bins=25, edgecolor="black")
                ax.set_xlabel("Total Match Points")
                ax.set_ylabel("Number of Matches")
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.subheader("Boxplot of Total Points")
                fig, ax = plt.subplots()
                sns.boxplot(x=points, ax=ax)
                ax.set_xlabel("Total Match Points")
                st.pyplot(fig)
                plt.close(fig)

    if {"team1_points", "team2_points"}.issubset(filtered.columns):
        scatter_data = filtered[
            ["team1_points", "team2_points"]
        ].apply(pd.to_numeric, errors="coerce").dropna()

        if not scatter_data.empty:
            st.subheader("Points Scored by Both Match Participants")
            fig, ax = plt.subplots()
            ax.scatter(
                scatter_data["team1_points"],
                scatter_data["team2_points"],
                alpha=0.4
            )
            ax.set_xlabel("Team 1 Points")
            ax.set_ylabel("Team 2 Points")
            st.pyplot(fig)
            plt.close(fig)

    if "games_played" in filtered.columns:
        games = pd.to_numeric(
            filtered["games_played"], errors="coerce"
        ).dropna()

        if not games.empty:
            st.subheader("Number of Games per Match")
            counts = games.value_counts().sort_index()
            st.bar_chart(counts)

    if "date" in filtered.columns:
        yearly = filtered.dropna(subset=["date"]).copy()

        if not yearly.empty:
            yearly["Year"] = yearly["date"].dt.year
            yearly_counts = yearly.groupby("Year").size()

            st.subheader("Matches by Year")
            st.line_chart(yearly_counts)

    st.subheader("Recorded Winner Position")
    winner_counts = filtered["winner"].map({
        1: "Team 1", 2: "Team 2"
    }).value_counts()

    st.bar_chart(winner_counts)

# ==================================================
# TAB 4: STATISTICAL ANALYSIS
# ==================================================

with statistics:
    st.header("Statistical Analysis")

    numeric = filtered.select_dtypes(include=np.number)
    numeric = numeric.drop(
        columns=["winner"], errors="ignore"
    )

    st.subheader("Descriptive Statistics")

    if not numeric.empty:
        st.dataframe(numeric.describe().T, use_container_width=True)

        st.write("**Variance**")
        st.dataframe(numeric.var().rename("Variance").to_frame())

        st.write("**Mode**")
        st.dataframe(numeric.mode().head(1).T.rename(
            columns={0: "Mode"}
        ))

    # Confidence interval and one-sample t-test
    if "total_match_points" in filtered.columns:
        values = pd.to_numeric(
            filtered["total_match_points"], errors="coerce"
        ).dropna()

        st.subheader("Confidence Interval and One-Sample t-Test")

        if len(values) >= 2:
            confidence = 0.95
            mean_value = values.mean()
            sem = stats.sem(values)
            lower, upper = stats.t.interval(
                confidence,
                df=len(values) - 1,
                loc=mean_value,
                scale=sem
            )

            st.write(
                f"**95% confidence interval for mean total points:** "
                f"{lower:.2f} to {upper:.2f}"
            )

            reference = st.number_input(
                "Hypothesized mean total points",
                value=70.0
            )

            if st.button("Run one-sample t-test"):
                t_stat, p_value = ttest_1samp(values, reference)
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.6g}")

                if p_value < 0.05:
                    st.write(
                        "Result: Reject the null hypothesis at the 5% level."
                    )
                else:
                    st.write(
                        "Result: Insufficient evidence to reject "
                        "the null hypothesis at the 5% level."
                    )

    # Correlation and covariance
    corr_cols = [
        c for c in [
            "team1_points", "team2_points",
            "total_match_points", "point_difference"
        ] if c in filtered.columns
    ]

    if len(corr_cols) >= 2:
        corr_data = filtered[corr_cols].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        st.subheader("Correlation and Covariance")

        if len(corr_data) >= 2:
            st.write("Pearson Correlation Matrix")
            st.dataframe(corr_data.corr(), use_container_width=True)

            fig, ax = plt.subplots()
            sns.heatmap(
                corr_data.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )
            st.pyplot(fig)
            plt.close(fig)

            st.write("Covariance Matrix")
            st.dataframe(corr_data.cov(), use_container_width=True)

            x_col = st.selectbox(
                "First variable for correlation",
                corr_cols,
                index=0,
                key="corr_x"
            )
            y_options = [c for c in corr_cols if c != x_col]
            y_col = st.selectbox(
                "Second variable for correlation",
                y_options,
                index=0,
                key="corr_y"
            )

            pair = filtered[[x_col, y_col]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            if len(pair) >= 3:
                if pair[x_col].nunique() > 1 and pair[y_col].nunique() > 1:
                    pr, pp = pearsonr(pair[x_col], pair[y_col])
                    sr, sp = spearmanr(pair[x_col], pair[y_col])

                    st.write(
                        f"**Pearson:** r = {pr:.4f}, p-value = {pp:.6g}"
                    )
                    st.write(
                        f"**Spearman:** rho = {sr:.4f}, p-value = {sp:.6g}"
                    )
                else:
                    st.info("Correlation requires variation in both variables.")

    # Chi-square test
    st.subheader("Chi-Square Test")

    if "games_played" in filtered.columns:
        chi_data = filtered[["winner", "games_played"]].copy()
        chi_data["games_played"] = pd.to_numeric(
            chi_data["games_played"], errors="coerce"
        )
        chi_data = chi_data.dropna()
        chi_data = chi_data[chi_data["games_played"].isin([2, 3])]

        if not chi_data.empty:
            contingency = pd.crosstab(
                chi_data["winner"], chi_data["games_played"]
            )

            st.write("Contingency Table")
            st.dataframe(contingency)

            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p, dof, expected = chi2_contingency(contingency)
                st.write(f"Chi-square statistic: {chi2:.4f}")
                st.write(f"Degrees of freedom: {dof}")
                st.write(f"p-value: {p:.6g}")
                st.caption(
                    "Check expected cell counts and independence "
                    "before interpreting the test."
                )
            else:
                st.info("Not enough categories for a chi-square test.")

    # Two-way ANOVA
    st.subheader("Two-Way ANOVA")

    if {"total_match_points", "games_played", "winner"}.issubset(filtered.columns):
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            anova_data = filtered[
                ["total_match_points", "games_played", "winner"]
            ].copy()

            anova_data["total_match_points"] = pd.to_numeric(
                anova_data["total_match_points"], errors="coerce"
            )
            anova_data["games_played"] = pd.to_numeric(
                anova_data["games_played"], errors="coerce"
            )

            anova_data = anova_data.dropna()
            anova_data = anova_data[
                anova_data["games_played"].isin([2, 3])
                & anova_data["winner"].isin([1, 2])
            ].copy()

            if (
                len(anova_data) >= 10
                and anova_data["games_played"].nunique() >= 2
                and anova_data["winner"].nunique() >= 2
            ):
                anova_data["games_played"] = (
                    anova_data["games_played"].astype(str)
                )
                anova_data["winner"] = anova_data["winner"].astype(str)

                model = ols(
                    "total_match_points ~ "
                    "C(games_played) * C(winner)",
                    data=anova_data
                ).fit()

                table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(table, use_container_width=True)
                st.caption(
                    "This is an exploratory analysis. Games played is "
                    "related to total points by construction, and "
                    "match-level observations may not be independent."
                )
            else:
                st.info("Not enough data across the required ANOVA groups.")
        except Exception as e:
            st.warning(f"ANOVA could not be calculated: {e}")

# ==================================================
# TAB 5: MACHINE LEARNING
# ==================================================

with machine_learning:
    st.header("Machine Learning Experiments")

    st.warning(
        "These models are educational experiments, not validated "
        "professional predictions. Historical results and match scoring "
        "have important dependencies and limitations."
    )

    st.subheader("Regression")

    regression_features = [
        c for c in ["team2_points", "games_played"]
        if c in filtered.columns
    ]

    if (
        "team1_points" in filtered.columns
        and regression_features
    ):
        reg_data = filtered[
            ["team1_points"] + regression_features
        ].copy()

        reg_data = reg_data.apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        reg_data = reg_data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(reg_data) >= 20:
            X = reg_data[regression_features]
            y = reg_data["team1_points"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            reg_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(max_iter=10000)
            }

            results = []

            for name, model in reg_models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                results.append({
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                    "R2": r2_score(y_test, predictions)
                })

            st.dataframe(
                pd.DataFrame(results).set_index("Model"),
                use_container_width=True
            )

            st.caption(
                "The target and features are match-level scoring "
                "variables. This is a regression demonstration, not "
                "a pre-match prediction."
            )
        else:
            st.info("At least 20 complete rows are needed for regression.")
    else:
        st.info("Regression requires team1_points and suitable features.")

    st.divider()
    st.subheader("Classification: Historical Win-Rate Features")

    if "date" not in filtered.columns or filtered["date"].isna().all():
        st.info("Classification requires valid match dates.")
    else:
        ml_data = filtered.dropna(
            subset=["date", "team1", "team2", "winner"]
        ).sort_values("date").copy()

        # Features use only earlier rows in the date-sorted data.
        # Matches on the same date may still have ordering limitations.
        history = {}
        rate1, rate2 = [], []

        for row in ml_data.itertuples(index=False):
            p1 = str(getattr(row, "team1"))
            p2 = str(getattr(row, "team2"))
            winner = getattr(row, "winner")

            n1, w1 = history.get(p1, (0, 0))
            n2, w2 = history.get(p2, (0, 0))

            rate1.append(w1 / n1 if n1 else 0.5)
            rate2.append(w2 / n2 if n2 else 0.5)

            if winner == 1:
                history[p1] = (n1 + 1, w1 + 1)
                history[p2] = (n2 + 1, w2)
            else:
                history[p1] = (n1 + 1, w1)
                history[p2] = (n2 + 1, w2 + 1)

        ml_data["team1_prior_win_rate"] = rate1
        ml_data["team2_prior_win_rate"] = rate2
        ml_data["target"] = (ml_data["winner"] == 1).astype(int)

        if len(ml_data) >= 50 and ml_data["target"].nunique() == 2:
            X = ml_data[
                ["team1_prior_win_rate", "team2_prior_win_rate"]
            ]
            y = ml_data["target"]

            split = int(len(ml_data) * 0.8)

            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            if (
                len(X_test) > 0
                and y_train.nunique() == 2
                and y_test.nunique() == 2
            ):
                models = {
                    "Decision Tree": DecisionTreeClassifier(
                        max_depth=5, random_state=42
                    ),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=100, max_depth=8, random_state=42,
                        n_jobs=-1
                    ),
                    "KNN": make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier(n_neighbors=5)
                    ),
                    "SVC": make_pipeline(
                        StandardScaler(),
                        SVC()
                    )
                }

                st.subheader("Model Comparison")
                st.write(
                    "Train all four classifiers on the same chronological "
                    "train/test split and compare their evaluation metrics."
                )

                if st.button("Compare All Models", key="compare_all_models"):
                    comparison_rows = []
                    progress = st.progress(0)
                    status = st.empty()

                    for i, (model_name, model_instance) in enumerate(models.items()):
                        try:
                            status.write(f"Training {model_name}...")
                            model_instance.fit(X_train, y_train)
                            model_predictions = model_instance.predict(X_test)

                            comparison_rows.append({
                                "Model": model_name,
                                "Accuracy": accuracy_score(y_test, model_predictions),
                                "Precision": precision_score(
                                    y_test, model_predictions, zero_division=0
                                ),
                                "Recall": recall_score(
                                    y_test, model_predictions, zero_division=0
                                ),
                                "F1-score": f1_score(
                                    y_test, model_predictions, zero_division=0
                                )
                            })
                        except Exception as e:
                            st.warning(f"{model_name} failed: {e}")

                        progress.progress((i + 1) / len(models))

                    status.empty()
                    progress.empty()

                    if comparison_rows:
                        comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
                        st.dataframe(
                            comparison_df.style.format("{:.3f}"),
                            use_container_width=True
                        )

                        metric_to_plot = st.selectbox(
                            "Metric to compare",
                            ["Accuracy", "Precision", "Recall", "F1-score"],
                            index=3,
                            key="comparison_metric"
                        )
                        st.bar_chart(comparison_df[metric_to_plot])

                        best_model = comparison_df["F1-score"].idxmax()
                        st.success(
                            f"Highest F1-score on this test set: {best_model} "
                            f"({comparison_df.loc[best_model, 'F1-score']:.3f})"
                        )
                        st.caption(
                            "This identifies the top result on this test split only; "
                            "it does not guarantee better performance on new data."
                        )

                st.divider()
                st.subheader("Train One Model in Detail")
                chosen = st.selectbox(
                    "Select classification model",
                    list(models.keys()),
                    key="single_classification_model"
                )

                if st.button("Train and evaluate classifier", key="train_classifier"):
                    model = models[chosen]

                    try:
                        with st.spinner(f"Training {chosen}... Please wait."):
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)

                        st.success(f"{chosen} trained successfully!")

                        metrics = {
                            "Accuracy": accuracy_score(y_test, predictions),
                            "Precision": precision_score(
                                y_test, predictions, zero_division=0
                            ),
                            "Recall": recall_score(
                                y_test, predictions, zero_division=0
                            ),
                            "F1-score": f1_score(
                                y_test, predictions, zero_division=0
                            )
                        }

                        cols = st.columns(4)
                        for col, (name, value) in zip(cols, metrics.items()):
                            col.metric(name, f"{value:.3f}")

                        st.subheader("Classification Report")
                        report = classification_report(
                            y_test,
                            predictions,
                            labels=[0, 1],
                            target_names=["Team 2 wins", "Team 1 wins"],
                            output_dict=True,
                            zero_division=0
                        )
                        st.dataframe(
                            pd.DataFrame(report).T,
                            use_container_width=True
                        )

                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(
                            y_test, predictions, labels=[0, 1]
                        )

                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            annot_kws={"size": 10},
                            xticklabels=["Predicted Team 2", "Predicted Team 1"],
                            yticklabels=["Actual Team 2", "Actual Team 1"],
                            ax=ax
                        )

                        ax.set_xlabel("Prediction", fontsize=10)
                        ax.set_ylabel("Actual result", fontsize=10)
                        ax.tick_params(axis="both", labelsize=9)

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)
                        plt.close(fig)

                    except Exception as e:
                        st.error(f"{chosen} failed to train: {e}")

            else:
                st.info(
                    "The chronological split does not contain both "
                    "outcome classes in training and testing. Try a "
                    "larger date range."
                )
        else:
            st.info(
                "Classification requires at least 50 valid matches "
                "and both winner outcomes."
            )

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "Educational data science project | Python • Pandas • "
    "SciPy • Scikit-learn • Streamlit"
)
