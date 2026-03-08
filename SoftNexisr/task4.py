import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.power import TTestIndPower



ab_data = pd.read_csv("ab_data.csv")

print("Dataset Shape:")
print(ab_data.shape)

print("\nFirst 5 Rows:")
print(ab_data.head())


print("\nDataset Info")
print(ab_data.info())

print("\nMissing Values")
print(ab_data.isnull().sum())

print("\nGroup Distribution")
print(ab_data['group'].value_counts())



conversion_summary = ab_data.groupby("group")["converted"].agg(
    visitors="count",
    conversions="sum"
)

conversion_summary["conversion_rate"] = (
    conversion_summary["conversions"] / conversion_summary["visitors"]
)

print("\nConversion Summary")
print(conversion_summary)


conv_old = ab_data[ab_data['group'] == 'control']['converted'].sum()
conv_new = ab_data[ab_data['group'] == 'treatment']['converted'].sum()

n_old = ab_data[ab_data['group'] == 'control'].shape[0]
n_new = ab_data[ab_data['group'] == 'treatment'].shape[0]

z_score, p_value = proportions_ztest(
    [conv_new, conv_old],
    [n_new, n_old],
    alternative='larger'
)

print("\nZ-Test Results")
print("Z Score:", z_score)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis: New design is better")
else:
    print("Fail to Reject Null Hypothesis")


ci_old = proportion_confint(conv_old, n_old, alpha=0.05)
ci_new = proportion_confint(conv_new, n_new, alpha=0.05)

print("\nConfidence Interval - Control:", ci_old)
print("Confidence Interval - Treatment:", ci_new)


rates = [
    conv_old / n_old,
    conv_new / n_new
]

labels = ["Control", "Treatment"]

plt.figure(figsize=(8,6))

plt.bar(labels, rates)

plt.title("Conversion Rate Comparison")
plt.ylabel("Conversion Rate")
plt.xlabel("Group")

plt.show()


plt.figure(figsize=(8,6))

plt.errorbar(
    x=[0,1],
    y=rates,
    yerr=[
        [rates[0] - ci_old[0], rates[1] - ci_new[0]],
        [ci_old[1] - rates[0], ci_new[1] - rates[1]]
    ],
    fmt='o',
    capsize=10
)

plt.xticks([0,1], labels)

plt.title("95% Confidence Interval for Conversion Rates")

plt.ylabel("Conversion Rate")

plt.grid(alpha=0.3)

plt.show()

if 'device' in ab_data.columns:

    contingency_table = pd.crosstab(
        ab_data['device'],
        ab_data['converted']
    )

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\nChi-Square Test")
    print("p-value:", p)

    if p < 0.05:
        print("Device affects conversion")
    else:
        print("No significant device effect")


if 'session_duration' in ab_data.columns:

    duration_control = ab_data[
        ab_data['group']=='control'
    ]['session_duration']

    duration_treatment = ab_data[
        ab_data['group']=='treatment'
    ]['session_duration']

    t_stat, p_val = stats.ttest_ind(
        duration_treatment,
        duration_control
    )

    print("\nT-Test Result")
    print("p-value:", p_val)

    if p_val < 0.05:
        print("Session duration differs significantly")
    else:
        print("No significant duration difference")


effect_size = 0.2
power = 0.8

analysis = TTestIndPower()

sample_size = analysis.solve_power(
    effect_size=effect_size,
    power=power,
    alpha=0.05
)

print("\nRequired Sample Size per Group:")
print(int(sample_size))


print("\nBusiness Insights")

print("Control Conversion Rate:", conv_old/n_old)
print("Treatment Conversion Rate:", conv_new/n_new)

if p_value < 0.05:
    print("New design should be rolled out")
else:
    print("New design does not significantly improve conversions")