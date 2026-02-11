import streamlit as st
import pandas as pd
import plotly.express as px
from schema_engine import SchemaEngine
from architect import GenericArchitect
from auditor import Auditor
from solver import SchedulingSolver
import json
import os
from datetime import date

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
    
    # 1. Add Plan Start Date Configuration
    st.subheader("Plan Parameters")
    # Default to a future date or current month. Jan 1, 2025 as per user example scenario
    plan_start_date = st.date_input("Plan Start Date (Month 1)", value=date(2025, 1, 1))

    st.divider()
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV (Any Schema)", type="csv")
    
    # Initialize whenever file is uploaded
    if uploaded_file:
        # Check if the file is different from the one in session state
        # Streamlit's file_uploader returns a new object on each rerun if the file changes
        # We can use a hash or just reload if it exists, as small overhead is acceptable here.

        # Simple fix: Always reload if uploaded_file is present.
        # Ideally, we check file_id or name, but for this demo, reloading is safer than stale data.
        df = pd.read_csv(uploaded_file)
        
        # Check if we need to update state (e.g. new file)
        # We compare shapes or just overwrite. Overwriting is safest.
        st.session_state.df = df
        
        # GENERATE PASSPORT
        st.session_state.passport = SchemaEngine.generate_passport(df)
        st.success("Schema Learned!")

    # Initialize Agents
    st.session_state.architect = GenericArchitect(HARDCODED_API_KEY)
    st.session_state.auditor = Auditor(HARDCODED_API_KEY)

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
                        st.session_state.constraints,
                        st.session_state.messages, # Pass history
                        plan_start_date=str(plan_start_date) # Pass configured date
                    )

                # 3. Handle Result
                if status == "PASS":
                    auditor_placeholder.success(f"Audit Passed: {message}")
                    
                    with st.spinner("Architecting Solution..."):
                        architect = st.session_state.architect
                        cdl, error = architect.generate_cdl(
                            prompt,
                            st.session_state.passport,
                            st.session_state.df,
                            st.session_state.messages, # Pass history
                            plan_start_date=str(plan_start_date) # Pass configured date
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

    # PANEL 3: EXECUTION & VISUALIZATION (Bottom)
    st.divider()
    st.subheader("Layer 3: The Universal Solver")

    if st.button("🚀 Generate Schedule"):
        if not st.session_state.constraints:
            st.warning("No constraints to solve!")
        else:
            with st.spinner("Optimizing Schedule..."):
                # Prepare Plan JSON
                plan_json = {
                    "global_parameters": {
                        "total_duration_months": 48,
                        "start_month_index": 1
                    },
                    "constraints": st.session_state.constraints
                }

                # Convert DF to list of dicts
                pod_data = st.session_state.df.to_dict('records')

                # Solve
                solver = SchedulingSolver(pod_data, plan_json)
                result_df = solver.solve()

                if result_df is not None:
                    st.session_state.result_df = result_df
                    st.success(f"Schedule Generated! ({len(result_df)} assignments)")
                else:
                    st.error("No Feasible Solution Found.")

    # Display Result
    if 'result_df' in st.session_state:
        res = st.session_state.result_df

        # 1. Visualization
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            # Aggregate per month
            monthly_counts = res['AssignedMonth'].value_counts().sort_index().reset_index()
            monthly_counts.columns = ['Month', 'Pod Count']
            fig = px.bar(monthly_counts, x='Month', y='Pod Count', title="Migration Wave Schedule")
            st.plotly_chart(fig, use_container_width=True)

        with col_v2:
            st.write("#### Download Plan")
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="universal_migration_plan.csv",
                mime="text/csv"
            )

        with st.expander("View Detailed Schedule"):
            st.dataframe(res, use_container_width=True)

else:
    st.info("Please upload a CSV to start.")
