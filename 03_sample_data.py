import pandas as pd
import sqlite3

conn = sqlite3.connect("fraud.db")
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# On garde toutes les fraudes (rares, précieuses) + un échantillon de transactions normales
fraud = df[df["Class"] == 1]
normal_sample = df[df["Class"] == 0].sample(n=5000, random_state=42)

df_sample = pd.concat([fraud, normal_sample]).sample(frac=1, random_state=42)  # mélange

conn2 = sqlite3.connect("fraud_sample.db")
df_sample.to_sql("transactions", conn2, if_exists="replace", index=False)
conn2.close()

print("Echantillon cree avec", len(df_sample), "lignes")