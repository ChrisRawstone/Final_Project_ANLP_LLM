import pandas as pd

df = pd.read_json("models_final/output_10_its/model5/nordjylland-news/df.json", lines=True)

print(df.head())
# Convert to CSV format
df.to_csv("models_final/output_10_its/model5/nordjylland-news/df_news.csv", index=False)

# Read back as CSV to verify
df_csv = pd.read_csv("models_final/output_10_its/model5/nordjylland-news/df.csv")
print("\nFirst few rows of CSV:")
print(df_csv.head())
