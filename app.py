import streamlit as st
import pandas as pd
import numpy as np
from draftsim import run_simulations
from projsim import projections, prepare_draft_results, simulate_team_projections, run_parallel_simulations

# Streamlit app
st.sidebar.title("Fantasy Football Simulator")

# Create a navigation sidebar
page = st.sidebar.selectbox("Select a Page", ["Draft Simulator", "Projection Simulator"])

if page == "Draft Simulator":
    st.title('Fantasy Football Draft Simulator')

    # Download link for sample CSV
    sample_csv_path = 'adp_sheet_test.csv'
    with open(sample_csv_path, 'rb') as file:
        sample_csv = file.read()

    st.download_button(
        label="Download sample CSV",
        data=sample_csv,
        file_name='adp_sheet_test.csv',
        mime='text/csv',
    )

    # File upload
    uploaded_file = st.file_uploader("Upload your ADP CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check if player_id exists, if not, create it
        if 'player_id' not in df.columns:
            df['player_id'] = df.index

        st.write("Data Preview:")
        st.dataframe(df.head())

        # Parameters for the simulation
        num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
        num_teams = st.number_input("Number of teams", min_value=2, value=6)
        num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
        team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.95)

        if st.button("Run Simulation"):
            all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)

            # Save the draft results to a DataFrame
            draft_results = []
            for sim_num, draft in enumerate(all_drafts):
                for team, players in draft.items():
                    result_entry = {
                        'Simulation': sim_num + 1,
                        'Team': team,
                    }
                    for i, player in enumerate(players):
                        result_entry.update({
                            f'Player_{i+1}_Name': player['name'],
                            f'Player_{i+1}_Position': player['position'],
                            f'Player_{i+1}_Team': player['team']
                        })
                    draft_results.append(result_entry)

            draft_results_df = pd.DataFrame(draft_results)

            # Display the results
            st.dataframe(draft_results_df)

            # Download link for the results
            csv = draft_results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Draft Results",
                data=csv,
                file_name='draft_results_with_team_stacking_and_positions.csv',
                mime='text/csv',
            )

elif page == "Projection Simulator":
    st.title('Projection Simulator')

    # File upload for draft results
    uploaded_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

    if uploaded_file is not None:
        draft_results_df = pd.read_csv(uploaded_file)

        st.write("Draft Results Data Preview:")
        st.dataframe(draft_results_df.head())

        # Number of simulations for projection
        num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

        if st.button("Run Projection Simulation"):
            # Create a projection lookup dictionary for quick access
            projection_lookup = {
                name: (projections[name]['proj'], projections[name]['projsd'])
                for name in projections
            }

            # Run simulations
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

            # Display the results
            st.dataframe(final_results)

            # Download link for the results
            csv = final_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Projection Results",
                data=csv,
                file_name='projection_results.csv',
                mime='text/csv',
            )
