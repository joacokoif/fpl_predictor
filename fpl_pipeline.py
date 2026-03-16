import requests
import pandas as pd
import numpy as np
import pulp
import lightgbm as lgb
from typing import List, Dict, Any
import time

# FPL API endpoints
BASE_URL = "https://fantasy.premierleague.com/api/"
BOOTSTRAP_STATIC = BASE_URL + "bootstrap-static/"
ELEMENT_SUMMARY = BASE_URL + "element-summary/{}/"

def fetch_fpl_data() -> Dict[str, Any]:
    """Fetches base data from the FPL API."""
    print("Fetching FPL data from bootstrap-static...")
    response = requests.get(BOOTSTRAP_STATIC)
    response.raise_for_status()
    return response.json()

def get_next_gameweek(api_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts information about the upcoming Gameweek."""
    events = api_data.get('events', [])
    for event in events:
        if event.get('is_next'):
            return {
                'id': event.get('id'),
                'name': event.get('name'),
                'deadline_time': event.get('deadline_time')
            }
    return None

def extract_players_data(api_data: Dict[str, Any]) -> pd.DataFrame:
    """Extracts relevant player base static data."""
    elements = api_data['elements']
    df = pd.DataFrame(elements)
    
    # Process basic types
    df['now_cost'] = df['now_cost'] / 10.0
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['position'] = df['element_type'].map(pos_map)
    df = df[df['status'] == 'a'] # Keep only available players initially
    
    return df

def fetch_player_history(player_ids: List[int], max_players=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the match history and upcoming fixtures for each player.
    To avoid rate limits, we add a tiny sleep. For demo purposes, we can limit max_players.
    """
    if max_players:
        player_ids = player_ids[:max_players]
        
    print(f"Fetching historical match data for {len(player_ids)} players (this may take a minute)...")
    
    history_rows = []
    fixtures_rows = []
    
    for count, pid in enumerate(player_ids, 1):
        if count % 100 == 0:
            print(f"  Processed {count}/{len(player_ids)} players...")
            
        url = ELEMENT_SUMMARY.format(pid)
        try:
            res = requests.get(url)
            if res.status_code == 200:
                data = res.json()
                
                # History (Past Matches)
                for h in data.get('history', []):
                    h['element'] = pid
                    history_rows.append(h)
                    
                # Fixtures (Upcoming Matches)
                for f in data.get('fixtures', []):
                    # Only take the very next fixture (Gameweek)
                    if f.get('event_name'):
                        f['element'] = pid
                        fixtures_rows.append(f)
                        break # Only need the immediate next one for prediction
        except Exception as e:
            print(f"Error fetching data for player {pid}: {e}")
            
        # time.sleep(0.05) # Be kind to the API
        
    hist_df = pd.DataFrame(history_rows)
    fixt_df = pd.DataFrame(fixtures_rows)
    return hist_df, fixt_df

def feature_engineering_historical(hist_df: pd.DataFrame, api_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates historical features for the ML model, primarily rolling 5 gameweek averages.
    Target variable (y) will be the points scored in the *current* row's match.
    Features (X) will be shifted so we predict current match using *past* data.
    """
    print("Engineering historical features for Machine Learning...")
    
    # Reconstruct FDR (Fixture Difficulty Rating) for past matches using team strengths
    teams_df = pd.DataFrame(api_data['teams'])
    strength_map = dict(zip(teams_df['id'], teams_df['strength']))
    hist_df['fixture_difficulty'] = hist_df['opponent_team'].map(strength_map).fillna(3)
    
    # Sort by player and kickoff time
    hist_df['kickoff_time'] = pd.to_datetime(hist_df['kickoff_time'])
    hist_df = hist_df.sort_values(by=['element', 'kickoff_time'])
    
    # Ensure numeric types
    numeric_cols = ['total_points', 'minutes', 'bps', 'influence', 'creativity', 'threat', 'ict_index']
    for col in numeric_cols:
         hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)

    # Calculate Rolling 5 Averages per player
    # We shift(1) right away so the rolling average strictly looks at prior games, NOT the target game.
    def calculate_rolling(group):
        group['rolling_points_5'] = group['total_points'].shift(1).rolling(5, min_periods=1).sum()
        group['rolling_minutes_5'] = group['minutes'].shift(1).rolling(5, min_periods=1).mean()
        group['rolling_ict_5'] = group['ict_index'].shift(1).rolling(5, min_periods=1).mean()
        return group
        
    hist_df = hist_df.groupby('element', group_keys=False).apply(calculate_rolling)
    
    # Drop rows where we couldn't even get 1 past game of history
    hist_df = hist_df.dropna(subset=['rolling_points_5'])
    return hist_df

def train_lgbm_model(train_df: pd.DataFrame):
    """Trains a LightGBM model to predict total_points."""
    print("Training LightGBM model on historical player data...")
    
    features = ['rolling_points_5', 'rolling_minutes_5', 'rolling_ict_5', 'fixture_difficulty']
    target = 'total_points'
    
    X = train_df[features]
    y = train_df[target]
    
    # Define dataset
    train_data = lgb.Dataset(X, label=y)
    
    # Hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

def predict_next_gw(model, df_players: pd.DataFrame, hist_df: pd.DataFrame, fixt_df: pd.DataFrame) -> pd.DataFrame:
    """Predicts xP for the upcoming Gameweek using the trained LightGBM model."""
    print("Predicting Next Gameweek xP...")
    
    # We need to construct the feature row for the NEXT game for each player
    # 1. Get their latest rolling stats from the absolute last row of their history
    latest_stats = hist_df.groupby('element').last().reset_index()
    
    # Wait, the rolling stats in hist_df were shifted for training. 
    # The actual stats BEFORE the next game would be calculated using the latest row's ACTUAL values as well.
    # To keep it simple, let's recalculate the exact rolling 5 of their last 5 actual matches:
    
    def get_latest_actual_rolling(group):
        return pd.Series({
            'rolling_points_5': pd.to_numeric(group['total_points'], errors='coerce').fillna(0).tail(5).sum(),
            'rolling_minutes_5': pd.to_numeric(group['minutes'], errors='coerce').fillna(0).tail(5).mean(),
            'rolling_ict_5': pd.to_numeric(group['ict_index'], errors='coerce').fillna(0).tail(5).mean()
        })
        
    latest_real_stats = hist_df.groupby('element').apply(get_latest_actual_rolling).reset_index()
    
    # 2. Get their upcoming fixture difficulty
    # If a player has no upcoming fixtures (e.g. blank gameweek), drop them
    if fixt_df.empty:
        df_players['predicted_xP'] = 0.0
        return df_players
        
    upcoming_fdr = fixt_df[['element', 'difficulty']].rename(columns={'difficulty': 'fixture_difficulty'})
    
    # 3. Merge
    predict_df = pd.merge(latest_real_stats, upcoming_fdr, on='element', how='inner')
    
    # 4. Predict
    features = ['rolling_points_5', 'rolling_minutes_5', 'rolling_ict_5', 'fixture_difficulty']
    
    if not predict_df.empty:
        predict_df['predicted_xP'] = model.predict(predict_df[features])
        # Shrink negatives to 0
        predict_df['predicted_xP'] = predict_df['predicted_xP'].clip(lower=0.0)
    else:
        predict_df['predicted_xP'] = []
        
    # 5. Merge back to main players dataframe
    df_players = pd.merge(df_players, predict_df[['element', 'predicted_xP']], left_on='id', right_on='element', how='left')
    df_players['predicted_xP'] = df_players['predicted_xP'].fillna(0.0) # Players with no history/fixture get 0
    
    return df_players

def optimize_team(df: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
    """Formulates a MILP knapsack problem to select 15 players maximizing xP."""
    print(f"Solving optimization problem with budget: £{budget}m ...")
    
    df = df.dropna(subset=['predicted_xP', 'now_cost', 'team', 'element_type'])
    
    player_ids = df['id'].tolist()
    xp = dict(zip(df['id'], df['predicted_xP']))
    costs = dict(zip(df['id'], df['now_cost']))
    teams = dict(zip(df['id'], df['team']))
    positions = dict(zip(df['id'], df['element_type'])) # 1=GK, 2=DEF, 3=MID, 4=FWD
    
    prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("player", player_ids, cat="Binary")
    
    prob += pulp.lpSum([xp[i] * player_vars[i] for i in player_ids]), "Total_Expected_Points"
    prob += pulp.lpSum([costs[i] * player_vars[i] for i in player_ids]) <= budget, "Budget"
    prob += pulp.lpSum([player_vars[i] for i in player_ids]) == 15, "Total_Players"
    
    prob += pulp.lpSum([player_vars[i] for i in player_ids if positions[i] == 1]) == 2, "GKs"
    prob += pulp.lpSum([player_vars[i] for i in player_ids if positions[i] == 2]) == 5, "DEFs"
    prob += pulp.lpSum([player_vars[i] for i in player_ids if positions[i] == 3]) == 5, "MIDs"
    prob += pulp.lpSum([player_vars[i] for i in player_ids if positions[i] == 4]) == 3, "FWDs"
    
    unique_teams = df['team'].unique()
    for team in unique_teams:
        prob += pulp.lpSum([player_vars[i] for i in player_ids if teams[i] == team]) <= 3, f"Team_{team}_Max"
        
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[status] != 'Optimal':
        print("Could not find an optimal solution. Check constraints.")
        return pd.DataFrame()
        
    selected_players = []
    for i in player_ids:
        if pulp.value(player_vars[i]) == 1.0:
            selected_players.append(df[df['id'] == i])
            
    optimal_team_df = pd.concat(selected_players, ignore_index=True)
    
    total_cost = sum([costs[i] for i in player_ids if pulp.value(player_vars[i]) == 1.0])
    total_xp = pulp.value(prob.objective)
    
    print("\n" + "="*50)
    print("O P T I M I Z A T I O N   C O M P L E T E")
    print(f"Total Predicted xP (by ML): {total_xp:.2f}")
    print(f"Total Squad Cost:   £{total_cost:.1f}m / £{budget}m")
    print("="*50 + "\n")
    
    return optimal_team_df

def print_squad(squad_df: pd.DataFrame):
    """Prints the selected squad cleanly grouped by position."""
    pos_order = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    squad_df['pos_order'] = squad_df['position'].map(pos_order)
    squad_df = squad_df.sort_values(by=['pos_order', 'predicted_xP'], ascending=[True, False])
    
    print(f"{'Position':<10} {'Player Name':<25} {'Team ID':<8} {'Cost (£m)':<12} {'xP':<8}")
    print("-" * 65)
    
    for _, row in squad_df.iterrows():
        name = str(row['web_name']).encode('ascii', 'ignore').decode('ascii')
        print(f"{row['position']:<10} {name:<25} {row['team']:<8} £{row['now_cost']:<11.1f} {row['predicted_xP']:.2f}")

def main():
    print("Starting FPL ML Pipeline...")
    
    # 1. Fetch static data and gameweek
    api_data = fetch_fpl_data()
    next_gw = get_next_gameweek(api_data)
    if next_gw:
        print(f"\n[INFO] Optimizando equipo para: {next_gw['name']} (GW {next_gw['id']})")
        print(f"[INFO] Deadline: {next_gw['deadline_time']}\n")
    
    df_players = extract_players_data(api_data)
    player_ids = df_players['id'].tolist()
    
    # 2. Extract historical match data
    # (To prevent this script taking 10 minutes on 800 players, we'll fetch the top 200 by ownership to train/predict on for this demo)
    df_players = df_players.sort_values(by='selected_by_percent', ascending=False)
    top_players = df_players['id'].tolist()[:300] 
    
    hist_df, fixt_df = fetch_player_history(top_players)
    
    # 3. Engineer Training Features
    train_df = feature_engineering_historical(hist_df, api_data)
    
    # 4. Train LightGBM Model
    lgb_model = train_lgbm_model(train_df)
    
    # 5. Predict Next GW xP
    df_players_predicted = predict_next_gw(lgb_model, df_players, hist_df, fixt_df)
    
    # 6. Mathematical Optimization (Knapsack)
    squad = optimize_team(df_players_predicted, budget=100.0)
    
    if not squad.empty:
        print_squad(squad)

if __name__ == "__main__":
    main()
