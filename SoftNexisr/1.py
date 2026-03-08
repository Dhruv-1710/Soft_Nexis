import pandas as pd
import numpy as np



df = pd.read_excel("customers-100.xlsx")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nDataset Shape")
print(df.shape)




duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Duplicates Removed")




df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

print("\nClean Column Names")
print(df.columns)




print("\nMissing Values")
print(df.isna().sum())

missing_percent = (df.isna().sum() / len(df)) * 100
print("\nMissing Percentage")
print(missing_percent)




threshold = 0.7 * len(df)

df = df.dropna(axis=1, thresh=threshold)

print("\nColumns after dropping high missing columns")
print(df.columns)




num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())




cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


print("\nMissing Values After Cleaning")
print(df.isna().sum())




for col in df.columns:
    if "date" in col or "time" in col:
        df[col] = pd.to_datetime(df[col], errors="coerce")




for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "").str.replace("$", ""),
                errors="ignore"
            )
        except:
            pass


print("\nUpdated Data Types")
print(df.dtypes)




for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()




df.replace({
    "m": "male",
    "f": "female",
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom"
}, inplace=True)




for col in num_cols:
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    
    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])




print("\nFinal Dataset Shape")
print(df.shape)

print("\nRemaining Missing Values")
print(df.isna().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

assert df.duplicated().sum() == 0




df.to_csv(r"D:\cleaned_customers_data.csv", index=False)

print("\nData Cleaning Completed Successfully")