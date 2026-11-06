
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    norm, binom, poisson, uniform, expon,
    chi2_contingency, chisquare, ttest_1samp, f_oneway
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

# =====================================================
# 1. PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="Road Defect Analysis",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Road Defect Severity Analysis")
st.write(
    "Statistical Analysis and Probabilistic Prediction "
    "Using Multi-Sensor Data"
)
st.caption("Academic dashboard using a synthetic dataset.")

CSV_FILE = "road_defect_severity_synthetic_dataset.csv"
SEVERITY_ORDER = ["Low", "Moderate", "High", "Critical"]


# =====================================================
# 2. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)


try:
    df = load_data()
except FileNotFoundError:
    st.error("CSV file must be in the same folder as app.py.")
    st.stop()


required_columns = [
    "road_type",
    "severity_class",
    "severity_score_0_to_10",
    "vibration_level",
    "pothole_depth_cm",
    "defect_width_cm",
    "vehicle_speed_kmh",
    "rainfall_mm",
    "traffic_density_vehicles_per_hour",
    "defect_type"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()


numeric_columns = list(
    df.select_dtypes(include=np.number).columns
)

# =====================================================
# 3. DASHBOARD FILTERS
# =====================================================

st.sidebar.header("Dashboard Filters")

roads = sorted(df["road_type"].dropna().unique())

severities = [
    s for s in SEVERITY_ORDER
    if s in df["severity_class"].dropna().unique()
]

selected_roads = st.sidebar.multiselect(
    "Select Road Type",
    roads,
    default=roads
)

selected_severities = st.sidebar.multiselect(
    "Select Severity",
    severities,
    default=severities
)

data = df[
    df["road_type"].isin(selected_roads)
    & df["severity_class"].isin(selected_severities)
].copy()

alpha = st.sidebar.selectbox(
    "Significance Level",
    [0.01, 0.05, 0.10],
    index=1
)

benchmark = st.sidebar.number_input(
    "Hypothesized Mean Severity",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

if data.empty:
    st.warning("No records match the selected filters.")
    st.stop()

scores = data["severity_score_0_to_10"].dropna()

severity_counts = (
    data["severity_class"]
    .value_counts()
    .reindex(SEVERITY_ORDER, fill_value=0)
)


def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# 4. DATASET OVERVIEW
# =====================================================

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(data))

c2.metric(
    "Average Severity",
    f"{scores.mean():.2f}/10"
)

c3.metric(
    "Average Vibration",
    f"{data['vibration_level'].mean():.2f}"
)

c4.metric(
    "High / Critical Defects",
    int(data["severity_class"].isin(["High", "Critical"]).sum())
)

with st.expander("View Dataset"):
    st.dataframe(data, width="stretch")

with st.expander("Dataset Information and Missing Values"):
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.isnull().sum().rename("Missing Values"))


# =====================================================
# PART A: ORIGINAL PROJECT GRAPHS
# =====================================================

st.divider()
st.header("PART A - Road Defect Analysis")


# 5. SEVERITY DISTRIBUTION

st.subheader("1. Defect Severity Distribution")

fig, ax = plt.subplots(figsize=(8, 4))

severity_counts.plot(kind="bar", ax=ax)

ax.set_xlabel("Severity Class")
ax.set_ylabel("Number of Records")
ax.set_title("Road Defects by Severity")
ax.tick_params(axis="x", rotation=0)

show_plot(fig)


# 6. DESCRIPTIVE STATISTICS

st.subheader("2. Descriptive Statistical Analysis")

if len(scores) > 0:

    mode_values = scores.mode()
    mode_text = (
        ", ".join(f"{v:.2f}" for v in mode_values.head(3))
        if len(mode_values) else "N/A"
    )

    sd = scores.std() if len(scores) > 1 else np.nan

    cv = (
        (sd / scores.mean()) * 100
        if pd.notna(sd) and scores.mean() != 0
        else np.nan
    )

    a, b, c, d, e, f = st.columns(6)

    a.metric("Mean", f"{scores.mean():.2f}")
    b.metric("Median", f"{scores.median():.2f}")
    c.metric("Mode", mode_text)
    d.metric("Variance", f"{scores.var():.2f}")
    e.metric("Std Deviation", f"{sd:.2f}")
    f.metric("Coefficient of Variation", f"{cv:.2f}%" if pd.notna(cv) else "N/A")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(scores, bins=20, edgecolor="black")

    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Severity Scores")

    show_plot(fig)


# 7. CORRELATION

st.subheader("3. Pearson Correlation Heatmap")

corr_columns = [
    c for c in numeric_columns
    if c in data.columns
]

corr_data = data[corr_columns].dropna(axis=1, how="all")

if len(corr_data.columns) >= 2:

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        corr_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Between Sensors and Severity")

    show_plot(fig)


# 8. BOX PLOT

st.subheader("4. Severity by Defect Type")

box_data = data[
    ["defect_type", "severity_score_0_to_10"]
].dropna()

fig, ax = plt.subplots(figsize=(9, 4))

sns.boxplot(
    data=box_data,
    x="defect_type",
    y="severity_score_0_to_10",
    ax=ax
)

ax.set_xlabel("Defect Type")
ax.set_ylabel("Severity Score")
ax.set_title("Severity by Defect Type")
ax.tick_params(axis="x", rotation=20)

show_plot(fig)


# 9. PROBABILITY

st.subheader("5. Probabilistic Severity Analysis")

probability = (
    data["severity_class"]
    .value_counts(normalize=True)
    .reindex(SEVERITY_ORDER, fill_value=0)
    * 100
)

p1, p2, p3, p4 = st.columns(4)

p1.metric("Low", f"{probability['Low']:.1f}%")
p2.metric("Moderate", f"{probability['Moderate']:.1f}%")
p3.metric("High", f"{probability['High']:.1f}%")
p4.metric("Critical", f"{probability['Critical']:.1f}%")

fig, ax = plt.subplots(figsize=(8, 4))

probability.plot(kind="bar", ax=ax)

ax.set_xlabel("Severity Class")
ax.set_ylabel("Percentage")
ax.set_title("Probability Distribution of Severity Classes")
ax.tick_params(axis="x", rotation=0)

show_plot(fig)

st.caption("Observed class proportions, not validated future predictions.")


# 10. ROAD TYPE

st.subheader("6. Severity by Road Type")

road_severity = pd.crosstab(
    data["road_type"],
    data["severity_class"]
).reindex(columns=SEVERITY_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(9, 5))

road_severity.plot(kind="bar", ax=ax)

ax.set_xlabel("Road Type")
ax.set_ylabel("Number of Defects")
ax.set_title("Severity Across Road Types")

show_plot(fig)


# 11. RAINFALL VS SEVERITY

st.subheader("7. Rainfall vs Severity")

rain_data = data[
    ["rainfall_mm", "severity_score_0_to_10"]
].dropna()

fig, ax = plt.subplots(figsize=(8, 4))

if len(rain_data) >= 2 and rain_data["rainfall_mm"].nunique() > 1:

    sns.regplot(
        data=rain_data,
        x="rainfall_mm",
        y="severity_score_0_to_10",
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax
    )

else:
    ax.scatter(
        rain_data["rainfall_mm"],
        rain_data["severity_score_0_to_10"]
    )

ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Severity Score")
ax.set_title("Rainfall and Road Severity")

show_plot(fig)


# 12. NORMAL DISTRIBUTION

st.subheader("8. Normal Distribution of Severity Scores")

if len(scores) >= 2:

    mean_value = scores.mean()
    std_value = scores.std()

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(
        scores,
        bins=20,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Observed Data"
    )

    if std_value > 0:

        x = np.linspace(scores.min(), scores.max(), 300)

        normal_curve = norm.pdf(
            x,
            mean_value,
            std_value
        )

        ax.plot(
            x,
            normal_curve,
            label="Fitted Normal Curve"
        )

    ax.axvline(
        mean_value,
        linestyle="--",
        label=f"Mean = {mean_value:.2f}"
    )

    ax.set_title("Severity Score Normal Distribution")
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Probability Density")
    ax.legend()

    show_plot(fig)


# 13. AVERAGE SEVERITY BY ROAD TYPE

st.subheader("9. Average Severity by Road Type")

average_severity = (
    data.groupby("road_type")["severity_score_0_to_10"]
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 4))

average_severity.plot(kind="bar", ax=ax)

ax.set_xlabel("Road Type")
ax.set_ylabel("Average Severity")
ax.set_title("Average Severity for Each Road Type")

show_plot(fig)


# 14. POTHOLE DEPTH

st.subheader("10. Pothole Depth vs Severity")

depth_data = data[
    ["pothole_depth_cm", "severity_score_0_to_10"]
].dropna()

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    depth_data["pothole_depth_cm"],
    depth_data["severity_score_0_to_10"],
    alpha=0.6
)

ax.set_xlabel("Pothole Depth (cm)")
ax.set_ylabel("Severity Score")
ax.set_title("Pothole Depth vs Severity")

show_plot(fig)


# =====================================================
# PART B: ADDITIONAL PNS CONCEPTS
# =====================================================

st.divider()
st.header("PART B - Additional Probability and Statistics Concepts")


# 15. QUARTILES AND IQR

st.subheader("11. Quartiles and Interquartile Range")

if len(scores) > 0:

    q = scores.quantile([0, 0.25, 0.50, 0.75, 1])

    st.write("Five Number Summary")

    st.dataframe(
        pd.DataFrame({
            "Measure": [
                "Minimum", "Q1", "Median", "Q3", "Maximum"
            ],
            "Value": q.values
        }),
        hide_index=True
    )

    iqr = q.iloc[3] - q.iloc[1]

    st.metric("Interquartile Range (IQR)", f"{iqr:.3f}")


# 16. SKEWNESS AND KURTOSIS

st.subheader("12. Skewness and Kurtosis")

if len(scores) >= 4:

    a, b = st.columns(2)

    a.metric("Skewness", f"{scores.skew():.3f}")
    b.metric("Kurtosis", f"{scores.kurtosis():.3f}")

    st.caption(
        "Skewness measures asymmetry. "
        "Kurtosis describes the distribution's tail behaviour."
    )


# 17. CONDITIONAL PROBABILITY AND BAYES THEOREM

st.subheader("13. Conditional Probability and Bayes' Theorem")

selected_road = st.selectbox(
    "Choose Road Type",
    roads,
    key="bayes_road"
)

selected_severity = st.selectbox(
    "Choose Severity",
    SEVERITY_ORDER,
    key="bayes_severity"
)

subset = data[data["road_type"] == selected_road]

if len(subset) > 0:

    conditional_probability = (
        (subset["severity_class"] == selected_severity).mean()
    )

    st.metric(
        "P(Severity | Road Type)",
        f"{conditional_probability:.4f}"
    )

    total_severity = (
        data["severity_class"] == selected_severity
    ).sum()

    if total_severity > 0:

        bayes_probability = (
            (
                (data["road_type"] == selected_road)
                & (data["severity_class"] == selected_severity)
            ).sum()
            / total_severity
        )

        st.metric(
            "P(Road Type | Severity)",
            f"{bayes_probability:.4f}"
        )


# 18. BERNOULLI AND BINOMIAL DISTRIBUTION

st.subheader("14. Bernoulli and Binomial Distribution")

p_high = float(
    data["severity_class"]
    .isin(["High", "Critical"])
    .mean()
)

n = st.slider("Number of Trials", 1, 100, 10)

k = st.slider(
    "Number of High/Critical Defects",
    0,
    n,
    min(3, n)
)

st.write(f"Observed probability p = {p_high:.4f}")

binomial_probability = binom.pmf(k, n, p_high)

st.metric(
    f"Binomial P(X = {k})",
    f"{binomial_probability:.6f}"
)

x = np.arange(n + 1)

fig, ax = plt.subplots(figsize=(8, 3))

ax.bar(x, binom.pmf(x, n, p_high))

ax.set_xlabel("Number of Events")
ax.set_ylabel("Probability")
ax.set_title("Binomial Distribution")

show_plot(fig)


# 19. POISSON DISTRIBUTION

st.subheader("15. Poisson Distribution")

st.info(
    "The dataset does not contain a defect rate per kilometre or "
    "per hour. Therefore, this is an illustrative theoretical example."
)

lam = st.number_input(
    "Assumed Poisson Rate (Lambda)",
    min_value=0.01,
    value=2.0
)

max_events = st.slider(
    "Maximum Events",
    5,
    25,
    12
)

x = np.arange(max_events + 1)

fig, ax = plt.subplots(figsize=(8, 3))

ax.bar(x, poisson.pmf(x, lam))

ax.set_xlabel("Number of Events")
ax.set_ylabel("Probability")
ax.set_title("Poisson Distribution")

show_plot(fig)


# 20. UNIFORM DISTRIBUTION

st.subheader("16. Uniform Distribution")

low = st.number_input(
    "Lower Limit",
    value=0.0
)

high = st.number_input(
    "Upper Limit",
    value=10.0
)

if high > low:

    x = np.linspace(low - 1, high + 1, 250)

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(
        x,
        uniform.pdf(x, loc=low, scale=high-low)
    )

    ax.set_title("Uniform Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")

    show_plot(fig)

else:
    st.warning("Upper limit must be greater than lower limit.")


# 21. EXPONENTIAL DISTRIBUTION

st.subheader("17. Exponential Distribution")

exp_mean = st.number_input(
    "Assumed Mean Interval",
    min_value=0.1,
    value=5.0
)

x = np.linspace(0, 5 * exp_mean, 250)

fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(
    x,
    expon.pdf(x, scale=exp_mean)
)

ax.set_title("Exponential Distribution")
ax.set_xlabel("Time / Interval")
ax.set_ylabel("Density")

show_plot(fig)


# 22. PMF, PDF AND CDF

st.subheader("18. PMF, PDF and CDF")

distribution_type = st.selectbox(
    "Select Distribution",
    [
        "Binomial PMF",
        "Poisson PMF",
        "Normal PDF and CDF"
    ]
)

fig, ax = plt.subplots(figsize=(8, 3))

if distribution_type == "Binomial PMF":

    x = np.arange(n + 1)
    ax.bar(x, binom.pmf(x, n, p_high))

elif distribution_type == "Poisson PMF":

    x = np.arange(max_events + 1)
    ax.bar(x, poisson.pmf(x, lam))

else:

    mean_value = scores.mean() if len(scores) else 5
    std_value = scores.std() if len(scores) > 1 else 1

    if std_value == 0:
        std_value = 1

    x = np.linspace(
        mean_value - 4 * std_value,
        mean_value + 4 * std_value,
        300
    )

    ax.plot(
        x,
        norm.pdf(x, mean_value, std_value),
        label="PDF"
    )

    ax.plot(
        x,
        norm.cdf(x, mean_value, std_value),
        label="CDF"
    )

    ax.legend()

ax.set_title(distribution_type)
ax.set_xlabel("Value")
ax.set_ylabel("Probability / Density")

show_plot(fig)


# 23. Z-SCORE

st.subheader("19. Z-Score Standardization")

if len(scores) >= 2 and scores.std() > 0:

    z_scores = (
        scores - scores.mean()
    ) / scores.std()

    st.write(
        "Z = (Observation - Mean) / Standard Deviation"
    )

    st.metric(
        "Values with |Z| greater than 2",
        int((z_scores.abs() > 2).sum())
    )

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.hist(z_scores, bins=20, edgecolor="black")
    ax.axvline(-2, linestyle="--")
    ax.axvline(2, linestyle="--")

    ax.set_title("Z-Score Distribution")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Frequency")

    show_plot(fig)


# 24. Z TEST AND T TEST

st.subheader("20. Z-Test and One-Sample T-Test")

st.write(
    f"Null Hypothesis: Mean severity = {benchmark:.2f}"
)

if len(scores) >= 2 and scores.std() > 0:

    standard_error = (
        scores.std() / np.sqrt(len(scores))
    )

    z_statistic = (
        (scores.mean() - benchmark) / standard_error
    )

    z_pvalue = 2 * norm.sf(abs(z_statistic))

    t_statistic, t_pvalue = ttest_1samp(
        scores,
        popmean=benchmark
    )

    a, b = st.columns(2)

    a.metric("Z Statistic", f"{z_statistic:.3f}")
    a.metric("Z Test p-value", f"{z_pvalue:.5f}")

    b.metric("T Statistic", f"{t_statistic:.3f}")
    b.metric("T Test p-value", f"{t_pvalue:.5f}")

    st.write(
        "Z-Test Decision: "
        + ("Reject H0" if z_pvalue < alpha else "Do not reject H0")
    )

    st.write(
        "T-Test Decision: "
        + ("Reject H0" if t_pvalue < alpha else "Do not reject H0")
    )

    st.caption(
        "The Z-test is approximate because population standard deviation "
        "is unknown. The t-test is generally preferred in that situation."
    )


# 25. CENTRAL LIMIT THEOREM

st.subheader("21. Central Limit Theorem")

if len(scores) >= 5:

    sample_size = st.slider(
        "Sample Size",
        2,
        min(100, len(scores)),
        min(10, len(scores))
    )

    repetitions = st.slider(
        "Number of Repeated Samples",
        100,
        2000,
        1000,
        step=100
    )

    rng = np.random.default_rng(42)

    values = scores.to_numpy()

    sample_means = np.array([
        rng.choice(
            values,
            sample_size,
            replace=True
        ).mean()
        for _ in range(repetitions)
    ])

    a, b = st.columns(2)

    a.metric(
        "Simulated Standard Error",
        f"{sample_means.std():.3f}"
    )

    b.metric(
        "Theoretical Standard Error",
        f"{scores.std() / np.sqrt(sample_size):.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.hist(
        sample_means,
        bins=30,
        density=True,
        edgecolor="black"
    )

    ax.set_title("Sampling Distribution of Sample Mean")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")

    show_plot(fig)


# 26. CONFIDENCE INTERVAL

st.subheader("22. Confidence Interval")

if len(scores) >= 2:

    confidence_interval = stats.t.interval(
        confidence=1-alpha,
        df=len(scores)-1,
        loc=scores.mean(),
        scale=stats.sem(scores)
    )

    st.write(
        f"{(1-alpha)*100:.0f}% Confidence Interval: "
        f"({confidence_interval[0]:.3f}, "
        f"{confidence_interval[1]:.3f})"
    )


# 27. CHI-SQUARE TEST OF INDEPENDENCE

st.subheader("23. Chi-Square Test of Independence")

st.write(
    "H0: Road type and severity class are independent."
)

contingency_table = pd.crosstab(
    data["road_type"],
    data["severity_class"]
).reindex(
    columns=SEVERITY_ORDER,
    fill_value=0
)

contingency_table = contingency_table.loc[
    contingency_table.sum(axis=1) > 0
]

test_table = contingency_table.loc[
    :,
    contingency_table.sum(axis=0) > 0
]

if test_table.shape[0] >= 2 and test_table.shape[1] >= 2:

    chi_stat, chi_p, chi_df, expected = chi2_contingency(
        test_table
    )

    a, b, c = st.columns(3)

    a.metric("Chi-Square Statistic", f"{chi_stat:.3f}")
    b.metric("Degrees of Freedom", chi_df)
    c.metric("p-value", f"{chi_p:.5f}")

    st.dataframe(contingency_table)

    st.write(
        "Decision: "
        + ("Reject H0" if chi_p < alpha else "Do not reject H0")
    )

    if (expected < 5).any():

        st.warning(
            "Some expected frequencies are below 5. "
            "The chi-square approximation may be unreliable."
        )

else:
    st.info(
        "Select at least two road types and two severity categories."
    )


# 28. CHI-SQUARE GOODNESS OF FIT

st.subheader("24. Chi-Square Goodness-of-Fit Test")

st.write(
    "Illustrative H0: All four severity classes are equally frequent."
)

observed = severity_counts.to_numpy()

if observed.sum() > 0:

    expected = np.repeat(
        observed.sum() / 4,
        4
    )

    gof_stat, gof_p = chisquare(
        observed,
        f_exp=expected
    )

    a, b = st.columns(2)

    a.metric("Chi-Square Statistic", f"{gof_stat:.3f}")
    b.metric("p-value", f"{gof_p:.5f}")

    st.caption(
        "Equal frequency is a demonstration hypothesis, "
        "not an assumption about actual road defects."
    )


# 29. ANOVA AND F TEST

st.subheader("25. One-Way ANOVA / F-Test")

st.write(
    "H0: Mean severity is equal across road types."
)

groups = [
    group["severity_score_0_to_10"].dropna().to_numpy()
    for _, group in data.groupby("road_type")
]

groups = [
    group for group in groups
    if len(group) >= 2
]

if len(groups) >= 2:

    f_stat, f_p = f_oneway(*groups)

    a, b = st.columns(2)

    a.metric("F Statistic", f"{f_stat:.3f}")
    b.metric("ANOVA p-value", f"{f_p:.5f}")

    st.write(
        "Decision: "
        + ("Reject H0" if f_p < alpha else "Do not reject H0")
    )


# 30. REGRESSION AND R-SQUARED

st.subheader("26. Linear Regression and R-Squared")

predictor_options = [
    "vibration_level",
    "pothole_depth_cm",
    "defect_width_cm",
    "vehicle_speed_kmh",
    "rainfall_mm",
    "traffic_density_vehicles_per_hour"
]

available_predictors = [
    c for c in predictor_options
    if c in data.columns
]

predictors = st.multiselect(
    "Select Predictor Variables",
    available_predictors,
    default=[
        c for c in [
            "pothole_depth_cm",
            "rainfall_mm",
            "vibration_level"
        ]
        if c in available_predictors
    ]
)

if predictors:

    regression_data = data[
        predictors + ["severity_score_0_to_10"]
    ].dropna()

    if len(regression_data) > len(predictors) + 2:

        X = regression_data[predictors]

        y = regression_data["severity_score_0_to_10"]

        model = LinearRegression()

        model.fit(X, y)

        predictions = model.predict(X)

        r2 = r2_score(y, predictions)

        rmse = np.sqrt(
            mean_squared_error(y, predictions)
        )

        adjusted_r2 = (
            1 - (1-r2) * (len(y)-1)
            / (len(y)-len(predictors)-1)
        )

        a, b, c = st.columns(3)

        a.metric("R-Squared", f"{r2:.4f}")
        b.metric("Adjusted R-Squared", f"{adjusted_r2:.4f}")
        c.metric("RMSE", f"{rmse:.3f}")

        st.write("Regression Coefficients")

        st.dataframe(
            pd.DataFrame({
                "Variable": predictors,
                "Coefficient": model.coef_
            }),
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.scatter(y, predictions, alpha=0.55)

        low = min(y.min(), predictions.min())
        high = max(y.max(), predictions.max())

        ax.plot(
            [low, high],
            [low, high],
            "--",
            color="red"
        )

        ax.set_title("Observed vs Fitted Severity")
        ax.set_xlabel("Observed Severity")
        ax.set_ylabel("Fitted Severity")

        show_plot(fig)

        ols_model = sm.OLS(
            y,
            sm.add_constant(X, has_constant="add")
        ).fit()

        with st.expander("View Regression Summary"):
            st.text(ols_model.summary().as_text())

        st.warning(
            "R-squared is calculated on the same records used for fitting. "
            "It does not establish future prediction accuracy. "
            "A separate test dataset is needed for that."
        )

    else:
        st.info("Not enough complete records for regression.")


# 31. SPEARMAN CORRELATION

st.subheader("27. Spearman Rank Correlation")

if len(available_predictors) >= 2:

    x_variable = st.selectbox(
        "First Variable",
        available_predictors,
        key="spearman_x"
    )

    y_variable = st.selectbox(
        "Second Variable",
        available_predictors,
        index=1,
        key="spearman_y"
    )

    rank_data = data[
        [x_variable, y_variable]
    ].dropna()

    if (
        len(rank_data) >= 3
        and rank_data[x_variable].nunique() > 1
        and rank_data[y_variable].nunique() > 1
    ):

        rho, p_value = stats.spearmanr(
            rank_data[x_variable],
            rank_data[y_variable]
        )

        a, b = st.columns(2)

        a.metric("Spearman Correlation", f"{rho:.3f}")
        b.metric("p-value", f"{p_value:.5f}")


# =====================================================
# FINAL FOOTER
# =====================================================

st.divider()

st.caption(
    "Road Defect Severity Analysis Dashboard | "
    "Synthetic Dataset | Probability and Statistics"
)
