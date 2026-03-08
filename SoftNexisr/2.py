import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = sns.load_dataset('titanic')

print("Data Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe(include='all'))



df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])



fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['age'], bins=30, kde=True, ax=ax[0]).set_title('Age Distribution')
sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]).set_title('Fare by Class')
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x='sex', hue='survived', data=df).set_title('Survival by Gender')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm').set_title('Correlation Heatmap')
plt.show()


pclass_survival = pd.crosstab(df['pclass'], df['survived'], normalize='index') * 100
print("\nSurvival % by Class:\n", pclass_survival)


z_scores = np.abs(stats.zscore(df['fare']))
outliers = df[z_scores > 3]
print(f"\nFound {len(outliers)} fare outliers")


g = sns.FacetGrid(df, col='survived', row='pclass', height=3)
g.map(sns.histplot, 'age', bins=20)
plt.show()


sns.pairplot(df[['age', 'fare', 'parch', 'survived']], hue='survived')
plt.show()