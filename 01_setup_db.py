import pandas as pd
import sqlite3

df = pd.read_csv("creditcard.csv")

print("Nombre de transactions :", len(df))
print("Nombre de fraudes :", df["Class"].sum())
print("Taux de fraude : {:.4f}%".format(100 * df["Class"].mean()))

conn = sqlite3.connect("fraud.db")
df.to_sql("transactions", conn, if_exists="replace", index=False)
conn.close()

print("Base créée avec succès")