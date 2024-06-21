import streamlit as st
import pandas as pd
from PIL import Image


def run():
    st.title("Titanic Dataset")

    image = Image.open("./res/header_img.jpg")
    st.image(image, use_column_width=True)

    df = pd.read_csv(r"././data/dataset.csv")

    st.dataframe(df)


if __name__ == "__main__":
    run()
