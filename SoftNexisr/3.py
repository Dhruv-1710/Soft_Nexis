import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('01-01-2021.csv')

df.drop(columns=['FIPS', 'Admin2'], inplace=True, errors='ignore')
df['Last_Update'] = pd.to_datetime(df['Last_Update'])


top_countries = df.groupby('Country_Region')['Confirmed'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='darkred')
plt.title('Top 10 Countries by Confirmed Cases (Jan 1, 2021)')
plt.ylabel('Confirmed Cases')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df[['Confirmed', 'Deaths', 'Recovered', 'Active']].corr(), annot=True, cmap='Reds')
plt.title('Cases Correlation Heatmap')
plt.show()


sns.jointplot(data=df, x='Incident_Rate', y='Case_Fatality_Ratio', kind='reg', color='orange')
plt.show()