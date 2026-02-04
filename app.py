import streamlit as st
import pandas as pd
from schema_engine import SchemaEngine
from architect import GenericArchitect
from auditor import Auditor
import json
import os

# --- HARDCODED CONFIGURATION ---
HARDCODED_API_KEY = "" 

st.set_page_config(layout="wide", page_title="Universal Planner Demo")

st.title("Universal Migration Planner: Schema-Aware CDL Generator")
st.markdown("""
This demo proves the system can adapt to **any** dataset schema and generate 
**Strict Mathematical CDL** without code changes.
""")

# --- STATE INITIALIZATION ---
if "constraints" not in st.session_state:
    st.session_state.constraints = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("Configuration")
    mock_mode = st.toggle("Enable Mock Mode", value=True)
    
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV (Any Schema)", type="csv")
    
    # Initialize whenever file is uploaded
    if uploaded_file:
        # Avoid reloading if same file (Streamlit handles this, but logic should be safe)
        if 'df' not in st.session_state: # Simplified check
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            # GENERATE PASSPORT
            st.session_state.passport = SchemaEngine.generate_passport(df)
            st.success("Schema Learned!")

    # Reload agents on every run to pick up config changes (like Mock Mode)
    st.session_state.architect = GenericArchitect(HARDCODED_API_KEY, mock_mode=mock_mode)
    st.session_state.auditor = Auditor(HARDCODED_API_KEY, mock_mode=mock_mode)

    st.divider()
    st.subheader("Active Constraints")
    if st.session_state.constraints:
        for i, c in enumerate(st.session_state.constraints):
            with st.expander(f"{i+1}. {c.get('description', 'Rule')}"):
                st.caption(f"ID: {c.get('rule_type')}")
                st.json(c)
    else:
        st.info("No constraints active.")

    if st.button("Reset Constraints"):
        st.session_state.constraints = []
        st.session_state.messages = []
        st.rerun()

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

    # PANEL 2: THE AGENTIC MIDDLEWARE
    with col2:
        st.subheader("Layer 2: The Agentic Middleware")
        st.info("Auditor (Logic Check) -> Architect (Math Generation)")
        
        # Display Chat History
        messages = st.session_state.get("messages", [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "json" in msg:
                    st.json(msg["json"])

        # User Input
        if prompt := st.chat_input("State your constraint..."):
            # 1. Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Audit Request
            with st.chat_message("assistant"):
                auditor_placeholder = st.empty()
                with st.spinner("Auditing request..."):
                    auditor = st.session_state.auditor
                    status, message = auditor.audit_request(
                        prompt,
                        st.session_state.passport, 
                        st.session_state.constraints
                    )

                # 3. Handle Result
                if status == "PASS":
                    auditor_placeholder.success(f"Audit Passed: {message}")
                    
                    with st.spinner("Architecting Solution..."):
                        architect = st.session_state.architect
                        cdl, error = architect.generate_cdl(
                            prompt,
                            st.session_state.passport,
                            st.session_state.df
                        )
                        
                        if error:
                            response = f"❌ **Generation Failed:** {error}"
                            st.error(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            response = f"✅ **Constraint Created:** {cdl.get('description')}"
                            st.markdown(response)
                            st.json(cdl)

                            # Save to State
                            st.session_state.constraints.append(cdl)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "json": cdl
                            })
                            st.rerun() # Refresh to update sidebar

                elif status == "CLARIFICATION_NEEDED":
                    response = f"❓ **Clarification Needed:** {message}"
                    auditor_placeholder.warning(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                else: # FAIL
                    response = f"⛔ **Request Rejected:** {message}"
                    auditor_placeholder.error(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload a CSV to start.")
