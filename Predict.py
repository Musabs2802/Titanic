import streamlit as st
import joblib
import pandas as pd
from PIL import Image

df = pd.read_csv("./data/dataset.csv")

st.cache_data.clear()

pclass_map = {1: "1st", 2: "2nd", 3: "3rd"}
embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton', None: 'None'}

def run():
    st.title("Would you have survived the Titanic?")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    st.sidebar.info("This app predicts whether you will be able to survive the Titanic or not")

    model = joblib.load("./output/gbmodel.sav")

    col1, col2 = st.columns(2)

    pclass = col1.selectbox("Ticket class", list(map(lambda x:pclass_map[x], df["Pclass"].unique())))
    sex = col2.selectbox("Sex", df["Sex"].unique())
    age = col1.slider("Age", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), step=1)
    fare = col2.slider("Fare", min_value=int(df["Fare"].min()), max_value=int(df["Fare"].max()))
    sibSp = col1.selectbox("No. of siblings / spouses", list(range(0, 10)))
    parch = col2.selectbox("No. of parents / children", list(range(0, 10)))
    embarked = col1.selectbox("Port of Embarkation", embarked_map.values())

    data = pd.DataFrame(
        [
            {
                "Pclass": list(pclass_map.keys())[list(pclass_map.values()).index(pclass)],
                "Sex": sex,
                "Age": age,
                "SibSp": sibSp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": list(embarked_map.keys())[list(embarked_map.values()).index(embarked)],
            }
        ]
    )

    oh_encoded_features = pd.get_dummies(data[["Sex", "Embarked"]], dtype="int")

    data_encoded = pd.concat([data, oh_encoded_features], axis=1)
    data_encoded = data_encoded.drop(columns=["Sex", "Embarked"], axis=1)

    data_encoded = data_encoded.reindex(
        columns=[
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_female",
            "Sex_male",
            "Embarked_C",
            "Embarked_Q",
            "Embarked_S",
        ],
        fill_value=0,
    )

    if st.button("Predict", type="primary"):
        output = model.predict(data_encoded)
        if output[0]:
            st.success("You will Survive :D")
        else:
            st.warning("You won't Survive :(")

if __name__ == "__main__":
    run()
