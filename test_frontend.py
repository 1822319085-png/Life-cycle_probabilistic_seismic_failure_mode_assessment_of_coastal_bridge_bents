import streamlit as st

st.set_page_config(page_title="Frontend Diagnostics", layout="centered")
st.title("Streamlit Frontend Diagnostics")
st.write("Streamlit version:", st.__version__)
st.html("<style>.deployment-check {color:#16803e;font-weight:700;}</style>")
st.html("<div class='deployment-check'>HTML/CSS rendering is working.</div>")
st.selectbox("Distribution", ("Normal", "Lognormal", "Uniform"))
st.number_input("Mean", value=2.0)
st.number_input("COV", value=0.10)
st.success("Python rendered all frontend test controls.")
