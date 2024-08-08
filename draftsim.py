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
    "Ja'Marr Chase": {'proj': 27, 'projsd': 9},
    "Justin Jefferson": {'proj': 26, 'projsd': 9},
    "Amon-Ra St. Brown": {'proj': 25, 'projsd': 8},
    "Bijan Robinson": {'proj': 24, 'projsd': 8},
    "Breece Hall": {'proj': 23, 'projsd': 8},
    "A.J. Brown": {'proj': 22, 'projsd': 8},
    "Puka Nacua": {'proj': 21, 'projsd': 8},
    "Garrett Wilson": {'proj': 20, 'projsd': 7},
    "Jahmyr Gibbs": {'proj': 19, 'projsd': 7},
    "Marvin Harrison": {'proj': 18, 'projsd': 7},
    "Drake London": {'proj': 17, 'projsd': 7},
    "Jonathan Taylor": {'proj': 16, 'projsd': 7},
    "Nico Collins": {'proj': 15, 'projsd': 7},
    "Chris Olave": {'proj': 14, 'projsd': 6},
    "Deebo Samuel": {'proj': 13, 'projsd': 6},
    "Saquon Barkley": {'proj': 12, 'projsd': 6},
    "Jaylen Waddle": {'proj': 11, 'projsd': 6},
    "Davante Adams": {'proj': 10, 'projsd': 6},
    "Brandon Aiyuk": {'proj': 9, 'projsd': 6},
    "De'Von Achane": {'proj': 8, 'projsd': 5},
    "Mike Evans": {'proj': 7, 'projsd': 5},
    "DeVonta Smith": {'proj': 6, 'projsd': 5},
    "DK Metcalf": {'proj': 6, 'projsd': 5},
    "Malik Nabers": {'proj': 6, 'projsd': 4},
    "Cooper Kupp": {'proj': 6, 'projsd': 4},
    "Kyren Williams": {'proj': 6, 'projsd': 4},
    "Derrick Henry": {'proj': 6, 'projsd': 4},
    "DJ Moore": {'proj': 6, 'projsd': 3},
    "Stefon Diggs": {'proj': 6, 'projsd': 3},
    "Michael Pittman Jr.": {'proj': 6, 'projsd': 3},
    "Tank Dell": {'proj': 6, 'projsd': 3},
    "Sam LaPorta": {'proj': 6, 'projsd': 3},
    "Zay Flowers": {'proj': 6, 'projsd': 3},
    "Josh Allen": {'proj': 6, 'projsd': 3},
    "Travis Kelce": {'proj': 6, 'projsd': 3},
    "George Pickens": {'proj': 6, 'projsd': 3},
    "Isiah Pacheco": {'proj': 6, 'projsd': 3},
    "Amari Cooper": {'proj': 6, 'projsd': 3},
    "Jalen Hurts": {'proj': 6, 'projsd': 3},
    "Tee Higgins": {'proj': 6, 'projsd': 3},
    "Travis Etienne Jr.": {'proj': 6, 'projsd': 3},
    "Patrick Mahomes": {'proj': 6, 'projsd': 3},
    "Christian Kirk": {'proj': 6, 'projsd': 3},
    "Trey McBride": {'proj': 6, 'projsd': 3},
    "Lamar Jackson": {'proj': 6, 'projsd': 3},
    "Mark Andrews": {'proj': 6, 'projsd': 3},
    "Terry McLaurin": {'proj': 6, 'projsd': 3},
    "Dalton Kincaid": {'proj': 6, 'projsd': 3},
    "Josh Jacobs": {'proj': 6, 'projsd': 3},
    "Hollywood Brown": {'proj': 6, 'projsd': 3},
    "Keenan Allen": {'proj': 6, 'projsd': 3},
    "James Cook": {'proj': 6, 'projsd': 3},
    "Anthony Richardson": {'proj': 6, 'projsd': 3},
    "Jayden Reed": {'proj': 6, 'projsd': 3},
    "Calvin Ridley": {'proj': 6, 'projsd': 3},
    "Chris Godwin": {'proj': 6, 'projsd': 3},
    "Rashee Rice": {'proj': 6, 'projsd': 3},
    "Keon Coleman": {'proj': 6, 'projsd': 3},
    "Kyler Murray": {'proj': 6, 'projsd': 3},
    "Aaron Jones": {'proj': 6, 'projsd': 3},
    "DeAndre Hopkins": {'proj': 6, 'projsd': 3},
    "Rhamondre Stevenson": {'proj': 6, 'projsd': 3},
    "James Conner": {'proj': 6, 'projsd': 3},
    "Najee Harris": {'proj': 6, 'projsd': 3},
    "Jameson Williams": {'proj': 6, 'projsd': 3},
    "Jake Ferguson": {'proj': 6, 'projsd': 3},
    "Jordan Addison": {'proj': 6, 'projsd': 3},
    "Curtis Samuel": {'proj': 6, 'projsd': 3},
    "Jaylen Warren": {'proj': 6, 'projsd': 3},
    "Zamir White": {'proj': 6, 'projsd': 3},
    "Joe Burrow": {'proj': 6, 'projsd': 3},
    "Jonathon Brooks": {'proj': 6, 'projsd': 3},
    "D'Andre Swift": {'proj': 6, 'projsd': 3},
    "Raheem Mostert": {'proj': 6, 'projsd': 3},
    "Dak Prescott": {'proj': 6, 'projsd': 3},
    "Courtland Sutton": {'proj': 6, 'projsd': 3},
    "Brock Bowers": {'proj': 6, 'projsd': 3},
    "Jordan Love": {'proj': 6, 'projsd': 3},
    "Zack Moss": {'proj': 6, 'projsd': 3},
    "Joshua Palmer": {'proj': 6, 'projsd': 3},
    "David Njoku": {'proj': 6, 'projsd': 3},
    "Tony Pollard": {'proj': 6, 'projsd': 3},
    "Jayden Daniels": {'proj': 6, 'projsd': 3},
    "Brian Robinson Jr.": {'proj': 6, 'projsd': 3},
    "Romeo Doubs": {'proj': 6, 'projsd': 3},
    "Rashid Shaheed": {'proj': 6, 'projsd': 3},
    "Tyler Lockett": {'proj': 6, 'projsd': 3},
    "Tyjae Spears": {'proj': 6, 'projsd': 3},
    "Chase Brown": {'proj': 6, 'projsd': 3},
    "Devin Singletary": {'proj': 6, 'projsd': 3},
    "Khalil Shakir": {'proj': 6, 'projsd': 3},
    "Brock Purdy": {'proj': 6, 'projsd': 3},
    "Javonte Williams": {'proj': 6, 'projsd': 3},
    "Caleb Williams": {'proj': 6, 'projsd': 3},
    "Dontayvion Wicks": {'proj': 6, 'projsd': 3},
    "Brandin Cooks": {'proj': 6, 'projsd': 3},
    "Dallas Goedert": {'proj': 6, 'projsd': 3},
    "Trey Benson": {'proj': 6, 'projsd': 3},
    "Trevor Lawrence": {'proj': 6, 'projsd': 3},
    "Gus Edwards": {'proj': 6, 'projsd': 3},
    "Jakobi Meyers": {'proj': 6, 'projsd': 3},
    "Blake Corum": {'proj': 6, 'projsd': 3},
    "Ezekiel Elliott": {'proj': 6, 'projsd': 3},
    "Jerry Jeudy": {'proj': 6, 'projsd': 3},
    "Tua Tagovailoa": {'proj': 6, 'projsd': 3},
    "Jared Goff": {'proj': 6, 'projsd': 3},
    "Adonai Mitchell": {'proj': 6, 'projsd': 3},
    "Jerome Ford": {'proj': 6, 'projsd': 3},
    "Nick Chubb": {'proj': 6, 'projsd': 3},
    "Ja'Lynn Polk": {'proj': 6, 'projsd': 3},
    "Pat Freiermuth": {'proj': 6, 'projsd': 3},
    "Austin Ekeler": {'proj': 6, 'projsd': 3},
    "Dalton Schultz": {'proj': 6, 'projsd': 3}
}

# Convert projections dictionary to a NumPy structured array
proj_dtype = np.dtype([('player_name', 'U50'), ('proj', 'f4'), ('projsd', 'f4')])
projections_array = np.array([(name, projections[name]['proj'], projections[name]['projsd']) for name in projections], dtype=proj_dtype)

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

if __name__ == '__main__':
    # Streamlit app
    st.title('Fantasy Football Draft Simulator')

    # Download link for sample CSV
    sample_csv_path = '/mnt/data/adp sheet test.csv'
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
            csv = draft_results_df.to_csv(index=False).encode('
