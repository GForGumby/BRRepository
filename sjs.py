import streamlit as st

# Set page title and layout
st.set_page_config(page_title="Fantasy Football Draft Simulator", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Draft Simulator", "Projections"])

# Load pages based on sidebar selection
if page == "Draft Simulator":
    import draft_simulator
    draft_simulator.show()
elif page == "Projections":
    import projections
    projections.show()
