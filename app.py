import streamlit as st
import pandas as pd
from schema_engine import SchemaEngine
from architect import GenericArchitect
import os

# --- HARDCODED CONFIGURATION ---
# Replace this string with your actual OpenAI API Key
HARDCODED_API_KEY = "" 

st.set_page_config(layout="wide", page_title="Universal Planner Demo")

st.title("Universal Migration Planner: Schema-Aware CDL Generator")
st.markdown("""
This demo proves the system can adapt to **any** dataset schema and generate 
**Strict Mathematical CDL** without code changes.
""")

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("Configuration")
    # API Key input removed. Using hardcoded key.
    
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV (Any Schema)", type="csv")
    
    # Initialize whenever file is uploaded (API Key is assumed present)
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Initialize Architect with hardcoded key
        if 'architect' not in st.session_state:
            st.session_state.architect = GenericArchitect(HARDCODED_API_KEY)
        
        # GENERATE PASSPORT
        st.session_state.passport = SchemaEngine.generate_passport(df)
        st.success("Schema Learned!")

# --- MAIN AREA ---
if 'df' in st.session_state:
    col1, col2 = st.columns([1, 1])

    # PANEL 1: THE EYES (Schema Context)
    with col1:
        st.subheader("Layer 1: The Schema Profiler")
        st.info("The Agent dynamically scans the file to understand the 'Physical Reality'.")
        with st.expander("View Data Passport (What the AI sees)", expanded=True):
            st.markdown(st.session_state.passport)
        
        st.dataframe(st.session_state.df.head(3), use_container_width=True)

    # PANEL 2: THE BRAIN (CDL Generation)
    with col2:
        st.subheader("Layer 2: The Architect")
        st.info("Converts intent into strict Operator/Operand trees.")
        
        user_input = st.text_area("Enter a Constraint:", 
                                  placeholder="Ex: 'Limit EMEA (LHR, FRA) to 50 pods in Jan' or 'Max 5 pods per Exadata'")
        
        if st.button("Generate CDL", type="primary"):
            if not user_input:
                st.warning("Please enter a constraint.")
            else:
                with st.spinner("Analyzing Schema & Generating Math..."):
                    architect = st.session_state.architect
                    # Pass the DF for JIT Validation
                    cdl, error = architect.generate_cdl(
                        user_input, 
                        st.session_state.passport, 
                        st.session_state.df 
                    )
                    
                    if error:
                        st.error(f"Validation Error: {error}")
                    else:
                        st.success("CDL Generated Successfully")
                        st.json(cdl)
                        
                        # EXPLAINER
                        st.markdown("### Why this is Generic:")
                        if 'params' in cdl and 'scope' in cdl['params']:
                             scope_val = cdl['params']['scope']
                        else:
                             scope_val = "GLOBAL"

                        st.markdown(f"""
                        - **Scope:** `{scope_val}` (Handles GroupBy)
                        - **Function:** Generic Primitives (SUM/COUNT)
                        - **Variables:** Mapped directly from Passport.
                        """)

else:
    st.info("Please upload a CSV to start.")