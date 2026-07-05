import streamlit as st
import pandas as pd
import sqlite3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection", layout="wide")

@st.cache_data
def load_data():
    conn = sqlite3.connect("fraud_sample.db")
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

df = load_data()
model, scaler, model_columns = load_model()

st.title("Detection de Fraude Bancaire")

tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Analyse detaillee", "Simulateur de prediction"])

with tab1:
    col1, col2, col3 = st.columns(3)
    fraud_rate = df["Class"].mean() * 100
    col1.metric("Taux de fraude", f"{fraud_rate:.3f}%")
    col2.metric("Transactions totales", f"{len(df):,}")
    col3.metric("Montant total fraude (EUR)", f"{df[df['Class']==1]['Amount'].sum():,.0f}")

    fig, ax = plt.subplots()
    df["Class"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Repartition Normal / Fraude")
    ax.set_xticklabels(["Normal", "Fraude"], rotation=0)
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Class", y="Amount", ax=ax, showfliers=False)
        ax.set_title("Montant des transactions : Normal vs Fraude")
        ax.set_xticklabels(["Normal", "Fraude"])
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df[df["Class"] == 1]["Amount"].hist(bins=30, ax=ax, color="#e74c3c")
        ax.set_title("Distribution des montants frauduleux")
        st.pyplot(fig)

with tab3:
    st.subheader("Predire si une transaction est frauduleuse")
    st.caption("Saisis les caracteristiques d'une transaction (valeurs V1-V28 issues d'une ACP, difficiles a interpreter individuellement)")

    amount = st.number_input("Montant de la transaction (EUR)", min_value=0.0, value=100.0)
    time = st.number_input("Temps depuis la premiere transaction (secondes)", min_value=0, value=50000)

    st.caption("Pour les variables V1 a V28, valeurs par defaut = 0 (transaction moyenne)")

    if st.button("Predire"):
        input_dict = dict.fromkeys(model_columns, 0.0)
        input_dict["Amount"] = amount
        input_dict["Time"] = time

        input_df = pd.DataFrame([input_dict])[model_columns]

        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"Transaction suspectee frauduleuse - probabilite : {proba*100:.2f}%")
        else:
            st.success(f"Transaction normale - probabilite de fraude : {proba*100:.2f}%")