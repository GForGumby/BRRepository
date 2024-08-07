import pandas as pd
import numpy as np
import streamlit as st

# Function to simulate a single draft
def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6, team_bonus=.95):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    # Initialize the teams
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    team_positions = {f'Team {i + starting_team_num}': {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0} for i in range(num_teams)}
    teams_stack = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    # Snake draft order
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                team_name = f'Team {pick_num + starting_team_num}'
                
                # Filter players based on positional requirements
                draftable_positions = []
                if team_positions[team_name]["QB"] < 1:
                    draftable_positions.append("QB")
                if team_positions[team_name]["RB"] < 1:
                    draftable_positions.append("RB")
                if team_positions[team_name]["WR"] < 2:
                    draftable_positions.append("WR")
                if team_positions[team_name]["TE"] < 1:
                    draftable_positions.append("TE")
                if team_positions[team_name]["FLEX"] < 1 and (team_positions[team_name]["RB"] + team_positions[team_name]["WR"] < 5):
                    draftable_positions.append("FLEX")
                
                df_filtered = df_copy.loc[
                    df_copy['position'].isin(draftable_positions) | 
                    ((df_copy['position'].isin(['RB', 'WR'])) & ('FLEX' in draftable_positions))
                ].copy()
                
                if df_filtered.empty:
                    continue
                
                # Adjust Simulated ADP based on team stacking
                df_filtered['Adjusted ADP'] = df_filtered.apply(
                    lambda x: x['Simulated ADP'] * team_bonus 
                    if x['team'] in teams_stack[team_name] else x['Simulated ADP'],
                    axis=1
                )
                
                df_filtered.sort_values('Adjusted ADP', inplace=True)
                
                selected_player = df_filtered.iloc[0]
                teams[team_name].append(selected_player)
                teams_stack[team_name].append(selected_player['team'])
                position = selected_player['position']
                if position in ["RB", "WR"]:
                    if team_positions[team_name][position] < {"RB": 1, "WR": 2}[position]:
                        team_positions[team_name][position] += 1
                    else:
                        team_positions[team_name]["FLEX"] += 1
                else:
                    team_positions[team_name][position] += 1
                df_copy = df_copy.loc[df_copy['player_id'] != selected_player['player_id']]
    
    return teams

# Function to run multiple simulations
def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=.95):
    all_drafts = []

    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds, team_bonus)
        all_drafts.append(draft_result)
    
    return all_drafts

# Streamlit app
st.title('Fantasy Football Draft Simulator')

# Download link for sample CSV
sample_csv_path = 'C:/Users/12013/Desktop/adp sheet test.csv'
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