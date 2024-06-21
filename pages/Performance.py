import streamlit as st
import joblib
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
from PIL import Image


def run():
    with open("./output/df_encoded.pkl", "rb") as f:
        df_encoded = pickle.load(f)

    st.title("Titanic Models Performance")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)


    model = joblib.load("./output/gbmodel.sav")

    df_encoded = df_encoded.sample(frac=1).reset_index(drop=True)
    split_index = int(len(df_encoded) * 0.8)
    test_df = df_encoded[split_index:]
    x_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]
    y_pred = model.predict(x_test)

    st.subheader("Accuracy Score:")
    st.text(accuracy_score(y_test, y_pred))

    st.subheader("Confusion Matrix:")
    st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))


if __name__ == "__main__":
    run()
