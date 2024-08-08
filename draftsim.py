import pandas as pd
import numpy as np
import streamlit as st
from numba import jit
from scipy.linalg import cholesky

# Define player projections and standard deviations
projections = {
    "Christian McCaffrey": {'proj': 30, 'projsd': 9},
    "CeeDee Lamb": {'proj': 29, 'projsd': 9},
    "Tyreek Hill": {'proj': 28, 'projsd': 9},
    # ... (continue with your full projections list)
    "Dalton Schultz": {'proj': 6, 'projsd': 3}
}

# JIT compiled function to generate projection
@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

# JIT compiled function to get payout based on rank
@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 20000.00
    elif rank == 2:
        return 6000.00
    elif rank == 3:
        return 3000.00
    elif rank == 4:
        return 1500.00
    elif rank == 5:
        return 1000.00
    elif rank == 6:
        return 500.00
    elif rank in [7, 8]:
        return 250.00
    elif rank in [9, 10]:
        return 200.00
    elif rank in range(11, 16):
        return 175.00
    elif rank in range(16, 21):
        return 150.00
    elif rank in range(21, 26):
        return 125.00
    elif rank in range(26, 36):
        return 100.00
    elif rank in range(36, 46):
        return 75.00
    elif rank in range(46, 71):
        return 60.00
    elif rank in range(71, 131):
        return 50.00
    elif rank in range(131, 251):
        return 40.00
    elif rank in range(251, 711):
        return 30.00
    else:
        return 0

# Function to prepare draft results in numpy array format
def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')
    player_positions = np.empty((num_teams, 6), dtype='U3')
    player_teams = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            draft_results[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Name']}"
            player_positions[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Position']}"
            player_teams[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Team']}"

    return draft_results, player_positions, player_teams, teams

# Function to create a simplified correlation matrix based on real-life NFL teams and positions
def create_correlation_matrix(player_teams, player_positions):
    num_players = player_teams.size
    correlation_matrix = np.identity(num_players)
    
    for i in range(num_players):
        for j in range(i + 1, num_players):
            if player_teams.flat[i] == player_teams.flat[j]:
                if player_positions.flat[i] == 'QB':
                    if player_positions.flat[j] == 'WR':
                        correlation_matrix[i, j] = 0.35
                        correlation_matrix[j, i] = 0.35
                    elif player_positions.flat[j] == 'TE':
                        correlation_matrix[i, j] = 0.25
                        correlation_matrix[j, i] = 0.25
                    elif player_positions.flat[j] == 'RB':
                        correlation_matrix[i, j] = 0.1
                        correlation_matrix[j, i] = 0.1
                elif player_positions.flat[j] == 'QB':
                    if player_positions.flat[i] == 'WR':
                        correlation_matrix[i, j] = 0.35
                        correlation_matrix[j, i] = 0.35
                    elif player_positions.flat[i] == 'TE':
                        correlation_matrix[i, j] = 0.25
                        correlation_matrix[j, i] = 0.25
                    elif player_positions.flat[i] == 'RB':
                        correlation_matrix[i, j] = 0.1
                        correlation_matrix[j, i] = 0.1

    return correlation_matrix

# Function to generate correlated projections
def generate_correlated_projections(player_names, player_positions, player_teams, projection_lookup, correlation_matrix):
    num_players = len(player_names)
    mean = np.array([projection_lookup[name][0] for name in player_names])
    std_dev = np.array([projection_lookup[name][1] for name in player_names])

    cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix
    L = cholesky(cov_matrix, lower=True)

    random_normals = np.random.normal(size=num_players)
    correlated_normals = np.dot(L, random_normals)
    correlated_projections = mean + correlated_normals

    return correlated_projections

# Function to simulate team projections from draft results
def simulate_team_projections(draft_results, player_positions, player_teams, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            team_player_names = draft_results[i]
            team_player_positions = player_positions[i]
            team_player_teams = player_teams[i]
            correlation_matrix = create_correlation_matrix(team_player_teams, team_player_positions)
            correlated_projections = generate_correlated_projections(team_player_names, team_player_positions, team_player_teams, projection_lookup, correlation_matrix)
            total_points[i] = np.sum(correlated_projections)

        # Rank teams
        ranks = total_points.argsort()[::-1].argsort() + 1

        # Assign payouts and accumulate them
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    # Calculate average payout per team
    avg_payouts = total_payouts / num_simulations
    return avg_payouts

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, player_positions, player_teams, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, player_positions, player_teams, projection_lookup, num_simulations)
    
    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit app
st.title('Fantasy Football Draft Simulator')

# Download link for sample CSV
sample_csv_path = 'adp sheet test.csv'
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

    # Additional code for projections and simulations if needed
    if st.button("Run Projections Simulation"):
        # Create a projection lookup dictionary for quick access
        projection_lookup = {
            name: (projections[name]['proj'], projections[name]['projsd'])
            for name in projections
        }

        # Run simulations
        final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        # Display the simulation results
        st.dataframe(final_results)

        # Download link for the simulation results
        csv_simulation = final_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Simulation Results",
            data=csv_simulation,
            file_name='simulation_results.csv',
            mime='text/csv',
        )

