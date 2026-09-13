# new_app2.py
import app_simulate
import Predictions
import streamlit as st

PAGES = {
    "Simulation": app_simulate,
    "Predictions": Predictions
}
st.sidebar.title('Navigation')
cols1, cols2 = st.columns((3,3,))

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
