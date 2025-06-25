import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("EV_Data.csv")
    return df

df = load_data()
st.title("EV Party Classifier ")
st.write("### Dataset Preview")
st.dataframe(df.head())

def preprocess(df):
    df = df.copy()
    df.dropna(subset=['Party'], inplace=True)

    num_cols = ['fuel_economy', 'gasoline_price_per_gallon']
    cat_cols = ['localofficials', 'affectweather', 'state']

    df = df[num_cols + cat_cols + ['Party']]

    le = LabelEncoder()
    df['Party'] = le.fit_transform(df['Party'])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop('Party', axis=1)
    y = df['Party']

    return X, y, le

X, y, le = preprocess(df)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_scaled, y_train)

st.write("### Enter EV Features to Predict Party")
user_input = {}

for col in ['fuel_economy', 'gasoline_price_per_gallon']:
    user_input[col] = st.number_input(f"{col}", value=10.0)

for col in ['localofficials', 'affectweather', 'state']:
    user_input[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

input_df = pd.DataFrame([user_input])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

input_scaled = sc.transform(input_df)

if st.button("Predict Party"):
    pred = rf_model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred])[0]
    st.success(f"🎯 Predicted Party: **{pred_label}**")
