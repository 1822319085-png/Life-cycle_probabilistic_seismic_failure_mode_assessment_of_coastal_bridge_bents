import streamlit as st

st.set_page_config(page_title="Markdown Frontend Diagnostics", layout="centered")
st.title("Markdown Frontend Diagnostics")
st.write("Streamlit version:", st.__version__)

# Deliberately avoid st.html, whose Html JavaScript module failed in the cloud test.
st.markdown("<style>.markdown-check {color:#16803e;font-weight:700;}</style>",
            unsafe_allow_html=True)
st.markdown("<div class='markdown-check'>Markdown-based HTML/CSS rendering test.</div>",
            unsafe_allow_html=True)
st.selectbox("Distribution", ("Normal", "Lognormal", "Uniform"))
st.number_input("Mean", value=2.0)
st.number_input("COV", value=0.10)
st.success("Python finished rendering all test controls.")
