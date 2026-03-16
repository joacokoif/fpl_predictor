import requests
import pandas as pd
import numpy as np
import pulp
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Any
import time
import argparse

# FPL API endpoints
BASE_URL = "https://fantasy.premierleague.com/api/"
BOOTSTRAP_STATIC = BASE_URL + "bootstrap-static/"
ELEMENT_SUMMARY = BASE_URL + "element-summary/{}/"
TEAM_PICKS = BASE_URL + "entry/{}/event/{}/picks/"

def get_current_gameweek(api_data: Dict[str, Any]) -> int:
    """Gets the ID of the most recently passed or currently active Gameweek."""
    events = api_data.get('events', [])
    for event in events:
        if event.get('is_current'):
            return event.get('id')
    for event in events:
         if event.get('is_previous'):
              return event.get('id')
    return 1 # Fallback

def fetch_team_from_id(team_id: int, current_gw: int) -> tuple[List[int], float, Dict[int, float]]:
    """Fetches a user's 15-man squad, bank, and purchase prices from the last Gameweek."""
    print(f"Fetching team data for FPL ID {team_id} (Gameweek {current_gw})...")
    url = TEAM_PICKS.format(team_id, current_gw)
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    player_ids = [pick['element'] for pick in data['picks']]
    
    # Store purchase prices (in £m) to correctly calculate selling values (Half-Profit rule)
    buy_prices = {pick['element']: pick['purchase_price'] / 10.0 for pick in data['picks'] if 'purchase_price' in pick}
    
    bank_budget_m = data['entry_history']['bank'] / 10.0
    return player_ids, bank_budget_m, buy_prices

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
    # Ensure starts and minutes exist even if missing
    if 'starts' not in df.columns:
        df['starts'] = 0
    if 'minutes' not in df.columns:
        df['minutes'] = 0
    # We must keep ALL players (even injured/unavailable) so that users can sell them.
    # The optimizer will naturally sell them if their xP is lower, or we can explicitly prevent buying them later.
    
    return df

def fetch_player_history(player_ids: List[int], next_n_fixtures: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the match history and upcoming fixtures for each player.
    """
    print(f"Fetching historical match data & {next_n_fixtures} upcoming fixtures for {len(player_ids)} players...")
    
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
                added_fixtures = 0
                for f in data.get('fixtures', []):
                    # Only take the next N fixtures
                    if f.get('event_name'):
                        f['element'] = pid
                        f['gw'] = f.get('event_name') # "Gameweek 28"
                        fixtures_rows.append(f)
                        added_fixtures += 1
                        if added_fixtures == next_n_fixtures:
                            break
        except Exception as e:
            print(f"Error fetching data for player {pid}: {e}")
            
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
    
    # Extract home away
    hist_df['was_home'] = hist_df['was_home'].astype(int)
    
    # Extract Positional Features mapping from main DF
    teams_df = pd.DataFrame(api_data['teams'])
    strength_map = dict(zip(teams_df['id'], teams_df['strength']))
    att_strength_map = dict(zip(teams_df['id'], teams_df['strength_attack_home']))
    def_strength_map = dict(zip(teams_df['id'], teams_df['strength_defence_home']))
    
    # FDR and opp strength
    hist_df['fixture_difficulty'] = hist_df['opponent_team'].map(strength_map).fillna(3)
    hist_df['opp_def_strength'] = hist_df['opponent_team'].map(def_strength_map).fillna(1100)
    hist_df['opp_att_strength'] = hist_df['opponent_team'].map(att_strength_map).fillna(1100)
    
    # Ensure numeric types for core raw stats
    numeric_cols = ['total_points', 'minutes', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'expected_goals', 'expected_assists']
    for col in numeric_cols:
         if col in hist_df.columns:
             hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)
         else:
             hist_df[col] = 0

    # Calculate Rolling 5 Averages per player
    def calculate_rolling(group):
        group['rolling_points_5'] = group['total_points'].shift(1).rolling(5, min_periods=1).sum()
        group['rolling_minutes_5'] = group['minutes'].shift(1).rolling(5, min_periods=1).mean()
        group['rolling_ict_5'] = group['ict_index'].shift(1).rolling(5, min_periods=1).mean()
        group['rolling_bps_5'] = group['bps'].shift(1).rolling(5, min_periods=1).mean()
        group['rolling_xG_5'] = group['expected_goals'].shift(1).rolling(5, min_periods=1).mean()
        group['rolling_xA_5'] = group['expected_assists'].shift(1).rolling(5, min_periods=1).mean()
        return group
        
    hist_df = hist_df.groupby('element', group_keys=False).apply(calculate_rolling)
    hist_df = hist_df.dropna(subset=['rolling_points_5'])
    return hist_df

def train_lgbm_multi_models(train_df: pd.DataFrame) -> Dict[str, lgb.Booster]:
    """Trains 5 separate LightGBM models for discrete FPL events using TimeSeriesSplit."""
    print("Training Multi-Model LightGBM Architecture on historical player data...")
    
    train_df = train_df.sort_values('kickoff_time')
    features = ['rolling_points_5', 'rolling_minutes_5', 'rolling_ict_5', 'rolling_bps_5', 'rolling_xG_5', 'rolling_xA_5', 'fixture_difficulty', 'opp_def_strength', 'opp_att_strength', 'was_home', 'element_type']
    
    # We map positions as an explicit feature too, as it vastly impacts clean sheets and goals
    if 'element_type' not in train_df.columns:
        train_df['element_type'] = 3 # default MID
        
    # Create the Discrete Event Targets
    train_df['target_played_60'] = (train_df['minutes'] >= 60).astype(int)
    train_df['target_played_sub'] = ((train_df['minutes'] > 0) & (train_df['minutes'] < 60)).astype(int)
    # FPL counts clean sheet if >= 60m and 0 goals conceded
    train_df['target_cs'] = ((train_df['minutes'] >= 60) & (train_df['clean_sheets'] > 0)).astype(int) if 'clean_sheets' in train_df.columns else 0
    train_df['target_goals'] = train_df['goals_scored'] if 'goals_scored' in train_df.columns else 0
    train_df['target_assists'] = train_df['assists'] if 'assists' in train_df.columns else 0

    X = train_df[features]
    
    models = {}
    
    # 1. Binary Classification Models (Logloss)
    binary_targets = ['target_played_60', 'target_played_sub', 'target_cs']
    binary_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 15, 'verbose': -1, 'random_state': 42
    }
    
    # 2. Poisson Regression Models for count events (rarer occurrences)
    poisson_targets = ['target_goals', 'target_assists']
    poisson_params = {
        'objective': 'poisson', 'metric': 'poisson', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 15, 'verbose': -1, 'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    def _train_target(target_name, params):
        best_model = None
        best_score = float('inf')
        y = train_df[target_name]
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(params, train_data, num_boost_round=150, valid_sets=[train_data, valid_data], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
            
            if params['objective'] == 'binary':
                from sklearn.metrics import log_loss
                preds = model.predict(X_test)
                # handle all zero test sets
                if len(np.unique(y_test)) > 1:
                    score = log_loss(y_test, preds)
                else:
                    score = np.mean(np.abs(preds - y_test))
            else:
                preds = model.predict(X_test)
                score = np.mean(np.abs(preds - y_test))
                
            if score < best_score:
                best_score = score
                best_model = model
                
        print(f"  - {target_name} trained (Best Fold Score: {best_score:.4f})")
        return best_model

    for bt in binary_targets:
        models[bt] = _train_target(bt, binary_params)
        
    for pt in poisson_targets:
        models[pt] = _train_target(pt, poisson_params)
        
    return models

def predict_5gw(models: Dict[str, lgb.Booster], df_players: pd.DataFrame, hist_df: pd.DataFrame, fixt_df: pd.DataFrame, api_data: Dict[str, Any]) -> pd.DataFrame:
    """Predicts multiple independent events for 5 gameweeks, reconstructing structural xP."""
    print("Predicting Multi-Event Probabilities for Next 5 Gameweeks...")
    
    def get_latest_actual_rolling(group):
        return pd.Series({
            'rolling_points_5': pd.to_numeric(group['total_points'], errors='coerce').fillna(0).tail(5).sum(),
            'rolling_minutes_5': pd.to_numeric(group['minutes'], errors='coerce').fillna(0).tail(5).mean(),
            'rolling_ict_5': pd.to_numeric(group['ict_index'], errors='coerce').fillna(0).tail(5).mean(),
            'rolling_bps_5': pd.to_numeric(group['bps'], errors='coerce').fillna(0).tail(5).mean(),
            'rolling_xG_5': pd.to_numeric(group.get('expected_goals'), errors='coerce').fillna(0).tail(5).mean(),
            'rolling_xA_5': pd.to_numeric(group.get('expected_assists'), errors='coerce').fillna(0).tail(5).mean(),
            'element_type': pd.to_numeric(group.get('element_type'), errors='coerce').fillna(3).tail(1).values[0] if 'element_type' in group.columns else 3
        })
        
    latest_real_stats = hist_df.groupby('element').apply(get_latest_actual_rolling).reset_index()
    
    # We patch element_type into latest_real_stats if it's not there, pulling from df_players
    if 'element_type' not in latest_real_stats.columns or latest_real_stats['element_type'].isna().all():
        pos_mapping = dict(zip(df_players['id'], df_players['element_type']))
        latest_real_stats['element_type'] = latest_real_stats['element'].map(pos_mapping).fillna(3)
    
    if fixt_df.empty:
        df_players['expected_points_5gw'] = 0.0
        return df_players, None, []
        
    upcoming_fdr = fixt_df[['element', 'difficulty', 'gw', 'is_home']].rename(columns={'difficulty': 'fixture_difficulty', 'is_home': 'was_home'})
    upcoming_fdr['was_home'] = upcoming_fdr['was_home'].astype(int)
    
    # Map opponent defense/attack to the upcoming fixture for full predictions
    teams_df = pd.DataFrame(api_data['teams']) if api_data else None
    
    # We will approximate opp strengths for upcoming if dict mappings are missing
    upcoming_fdr['opp_def_strength'] = upcoming_fdr['fixture_difficulty'] * 350
    upcoming_fdr['opp_att_strength'] = upcoming_fdr['fixture_difficulty'] * 350
    
    # Identify the chronologically NEXT gameweek name
    next_gw_name = fixt_df['gw'].iloc[0] if not fixt_df.empty else None
    
    import re
    def _extract_gw_num(gw_str):
        nums = re.findall(r'\d+', str(gw_str))
        return int(nums[0]) if nums else 0
        
    gw_list = sorted(fixt_df['gw'].unique().tolist(), key=_extract_gw_num)
    
    # Merge latest stats onto EACH of the upcoming fixtures
    predict_df = pd.merge(latest_real_stats, upcoming_fdr, on='element', how='inner')
    
    features = ['rolling_points_5', 'rolling_minutes_5', 'rolling_ict_5', 'rolling_bps_5', 'rolling_xG_5', 'rolling_xA_5', 'fixture_difficulty', 'opp_def_strength', 'opp_att_strength', 'was_home', 'element_type']
    
    if not predict_df.empty:
        # 1. Predict Individual Event Probabilities
        predict_df['p_60m'] = models['target_played_60'].predict(predict_df[features])
        predict_df['p_sub'] = models['target_played_sub'].predict(predict_df[features])
        predict_df['p_cs']  = models['target_cs'].predict(predict_df[features])
        predict_df['exp_g'] = models['target_goals'].predict(predict_df[features])
        predict_df['exp_a'] = models['target_assists'].predict(predict_df[features])
        
        # EXPERT FIX: Poisson models in LightGBM with small datasets can over-shrink. 
        # We blend in rolling averages to stabilize out-of-sample event predictions.
        predict_df['exp_g'] = (predict_df['exp_g'] * 0.5) + (predict_df['rolling_xG_5'] * 0.5)
        predict_df['exp_a'] = (predict_df['exp_a'] * 0.5) + (predict_df['rolling_xA_5'] * 0.5)

        # Baseline Points based on Playing Probability (Appearance Points)
        # FPL Rule: 2 pts for >= 60m, 1 pt for >0 and <60m
        app_points = (predict_df['p_60m'] * 2.0) + (predict_df['p_sub'] * 1.0)
        
        # Clean Sheet Points based on Position (1=GK, 2=DEF, 3=MID, 4=FWD)
        # FPL Rule: CS only awarded if played 60+ mins. 4 pts for GK/DEF, 1 pt for MID, 0 for FWD
        cs_multiplier = predict_df['element_type'].map({1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}).fillna(0.0)
        cs_points = predict_df['p_cs'] * predict_df['p_60m'] * cs_multiplier
        
        # Goals Scored Points
        # FPL Rule: GK/DEF=6, MID=5, FWD=4
        goal_multiplier = predict_df['element_type'].map({1: 6.0, 2: 6.0, 3: 5.0, 4: 4.0}).fillna(4.0)
        goal_points = predict_df['exp_g'] * goal_multiplier
        
        # Assists Points (3 pts for everyone)
        assist_points = predict_df['exp_a'] * 3.0
        
        # Bonus Points approximation (we can add a tiny fraction based on rolling BPS)
        # E.g. ~0.10 xP per BPS point above 15
        bonus_points = np.maximum(0, (predict_df['rolling_bps_5'] - 15) * 0.10) * predict_df['p_60m']
        
        # 2. Structural Reconstruction of xP
        predict_df['fixture_xP'] = app_points + cs_points + goal_points + assist_points + bonus_points
        predict_df['fixture_xP'] = predict_df['fixture_xP'].clip(lower=0.0)
        
        # Sum predictions for the 5-GW horizon
        sum_xp = predict_df.groupby('element')['fixture_xP'].sum().reset_index()
        sum_xp.rename(columns={'fixture_xP': 'expected_points_5gw'}, inplace=True)
        
        # We also need GW-specific predictions
        pivot_xp = predict_df.pivot_table(index='element', columns='gw', values='fixture_xP', aggfunc='sum', fill_value=0.0).reset_index()
    else:
        sum_xp = pd.DataFrame(columns=['element', 'expected_points_5gw'])
        pivot_xp = pd.DataFrame(columns=['element'] + gw_list)
        
    df_players = pd.merge(df_players, sum_xp, left_on='id', right_on='element', how='left')
    df_players = pd.merge(df_players, pivot_xp, left_on='id', right_on='element', how='left')
    
    df_players['expected_points_5gw'] = df_players['expected_points_5gw'].fillna(0.0)
    for gw in gw_list:
        if gw in df_players.columns:
            df_players[gw] = df_players[gw].fillna(0.0)
        else:
            df_players[gw] = 0.0
    
    return df_players, next_gw_name, gw_list

def optimize_transfers(df: pd.DataFrame, current_team_ids: List[int], bank_budget: float, initial_transfers: int, gw_list: List[str], top_n: int = 5, buy_prices: Dict[int, float] = None):
    """Multi-Period MILP: Optimizes transfers using pure ML xP, Auto-Subs, and exact FPL math."""
    if buy_prices is None:
        buy_prices = {}
        
    print(f"\nSolving multi-period optimization problem (Initial Bank: £{bank_budget}m, Initial FTs: {initial_transfers})...")
    
    # We require the GW expected points to exist
    for gw in gw_list:
        if gw not in df.columns:
            df[gw] = 0.0
            
    df = df.dropna(subset=['now_cost', 'team', 'element_type'])
    
    # Probability base (used only for Auto-Sub logic, NOT modifying raw base xP)
    if 'chance_of_playing_next_round' not in df.columns:
        df['chance_of_playing_next_round'] = 100
    df['play_prob'] = df['chance_of_playing_next_round'].fillna(100) / 100.0
    
    player_ids = df['id'].tolist()
    # The ML model already learned the true expected points intrinsically, so we take it cleanly.
    xp = {pid: [max(0.0, df[df['id'] == pid][gw].values[0]) for gw in gw_list] for pid in player_ids}
        
    costs = dict(zip(df['id'], df['now_cost']))
    teams = dict(zip(df['id'], df['team']))
    positions = dict(zip(df['id'], df['element_type']))
    probs = dict(zip(df['id'], df['play_prob']))
    unique_teams = df['team'].unique()
    
    current_ids_set = set(current_team_ids)
    
    # The true FPL Half-Profit Selling Rule
    # "For every 0.2m rise in price, you get 0.1m when selling."
    sell_values = {}
    for pid in player_ids:
        cur_price = costs[pid]
        if pid in current_ids_set and pid in buy_prices:
            bp = buy_prices[pid]
            sell_values[pid] = bp + np.floor((cur_price - bp) * 10 / 2) / 10.0 if cur_price > bp else cur_price
        else:
            sell_values[pid] = cur_price
            
    # Initial dynamic bank reference
    squad_sell_value = sum(sell_values[i] for i in current_team_ids)
    total_budget = squad_sell_value + bank_budget
    print(f"Current Squad Value: £{squad_sell_value:.1f}m | Total Available Budget: £{total_budget:.1f}m\n")
    
    prob = pulp.LpProblem("FPL_MultiPeriod_Optimization", pulp.LpMaximize)
    
    num_gws = len(gw_list)
    T = range(num_gws)
    
    # Decision Variables
    # s[i, t]: Player i is in the squad at Gameweek t (0 to 4)
    s = pulp.LpVariable.dicts("squad", (player_ids, T), cat="Binary")
    
    # y[i, t]: Player i is in the STARTING XI at Gameweek t
    y = pulp.LpVariable.dicts("start", (player_ids, T), cat="Binary")
    
    # c[i, t]: Player i is the CAPTAIN at Gameweek t
    c = pulp.LpVariable.dicts("cap", (player_ids, T), cat="Binary")
    
    # b[i, t]: Player i is BOUGHT prior to Gameweek t
    b = pulp.LpVariable.dicts("buy", (player_ids, T), cat="Binary")
    
    # sell[i, t]: Player i is SOLD prior to Gameweek t
    sell = pulp.LpVariable.dicts("sell", (player_ids, T), cat="Binary")
    
    # ft_avail[t]: Free transfers available at the start of Gameweek t
    ft_avail = pulp.LpVariable.dicts("ft_avail", T, lowBound=1, upBound=5, cat="Integer")
    # ft_carried[t]: Free transfers rolled over to the next week at the end of GW t
    ft_carried = pulp.LpVariable.dicts("ft_carried", T, lowBound=0, upBound=5, cat="Integer")
    # hits[t]: Extra transfers incurring a -4 penalty in GW t
    hits = pulp.LpVariable.dicts("hits", T, lowBound=0, cat="Integer")
    # is_hit[t]: Binary trigger for exact math
    is_hit = pulp.LpVariable.dicts("is_hit", T, cat="Binary")
    
    # Dynamic Bank variable: available money at GW t
    bank = pulp.LpVariable.dicts("bank", T, lowBound=0)
    
    # 1. State Transitions (Squad updating)
    for i in player_ids:
        for t in T:
            if t == 0:
                initial_presence = 1 if i in current_ids_set else 0
                prob += s[i][t] == initial_presence + b[i][t] - sell[i][t], f"Transition_{i}_{t}"
                prob += b[i][t] + sell[i][t] <= 1, f"No_Buy_Sell_{i}_{t}"
            else:
                prob += s[i][t] == s[i][t-1] + b[i][t] - sell[i][t], f"Transition_{i}_{t}"
                prob += b[i][t] + sell[i][t] <= 1, f"No_Buy_Sell_{i}_{t}"
                
    # 2. Perfect FT Linear Math & Dynamic Bank
    for t in T:
        transfers_made = pulp.lpSum([b[i][t] for i in player_ids])
        
        # Free transfer accumulation (capped at 5)
        if t == 0:
            prob += ft_avail[t] == initial_transfers, f"Init_FT_{t}"
        else:
            prob += ft_avail[t] == ft_carried[t-1] + 1, f"Avail_FT_Eq_{t}"
            prob += ft_avail[t] <= 5, f"Avail_FT_Max_{t}"
            
        prob += hits[t] >= transfers_made - ft_avail[t], f"Calc_Hits_Min_{t}"
        
        # Exactly mapping `hits = max(0, transfers - ft_avail)` 
        # using `is_hit` binary variable to avoid exploit loop holes
        prob += transfers_made - ft_avail[t] <= 15 * is_hit[t], f"Link_Hit_A_{t}"
        prob += transfers_made - ft_avail[t] >= -15 * (1 - is_hit[t]) + 0.1 - 15, f"Link_Hit_B_{t}" 
        
        prob += ft_carried[t] <= ft_avail[t] - transfers_made + 5 * is_hit[t], f"CarryOver_A_{t}"
        prob += ft_carried[t] <= 5 * (1 - is_hit[t]), f"CarryOver_B_{t}"
        prob += ft_carried[t] >= ft_avail[t] - transfers_made - 5 * is_hit[t], f"CarryOver_C_{t}"
        
        # Dynamic Bank Evolution (Using true sell_values)
        bought_cost = pulp.lpSum([costs[i] * b[i][t] for i in player_ids])
        sold_value = pulp.lpSum([sell_values[i] * sell[i][t] for i in player_ids])
        if t == 0:
            prob += bank[t] == bank_budget + sold_value - bought_cost, f"Bank_Evolution_{t}"
        else:
            prob += bank[t] == bank[t-1] + sold_value - bought_cost, f"Bank_Evolution_{t}"

    # 3. Global Week-by-Week Constraints
    for t in T:
        # Instead of static total_budget <= limit, explicitly the dynamic bank was already bounded to >= 0

        
        # 15 Players Total
        prob += pulp.lpSum([s[i][t] for i in player_ids]) == 15, f"Total_Players_{t}"
        
        # 11 Starters
        prob += pulp.lpSum([y[i][t] for i in player_ids]) == 11, f"Total_Starters_{t}"
        
        # 1 Captain
        prob += pulp.lpSum([c[i][t] for i in player_ids]) == 1, f"Total_Captain_{t}"
        
        for i in player_ids:
            prob += y[i][t] <= s[i][t], f"Starter_In_Squad_{i}_{t}"
            prob += c[i][t] <= y[i][t], f"Captain_Is_Starter_{i}_{t}"
        
        # Position Constraints (Squad)
        prob += pulp.lpSum([s[i][t] for i in player_ids if positions[i] == 1]) == 2, f"GKs_Squad_{t}"
        prob += pulp.lpSum([s[i][t] for i in player_ids if positions[i] == 2]) == 5, f"DEFs_Squad_{t}"
        prob += pulp.lpSum([s[i][t] for i in player_ids if positions[i] == 3]) == 5, f"MIDs_Squad_{t}"
        prob += pulp.lpSum([s[i][t] for i in player_ids if positions[i] == 4]) == 3, f"FWDs_Squad_{t}"
        
        # Position Constraints (Starting XI valid formations)
        prob += pulp.lpSum([y[i][t] for i in player_ids if positions[i] == 1]) == 1, f"GKs_Start_{t}"
        
        defs_start = pulp.lpSum([y[i][t] for i in player_ids if positions[i] == 2])
        prob += defs_start >= 3, f"DEFs_Start_Min_{t}"
        prob += defs_start <= 5, f"DEFs_Start_Max_{t}"
        
        mids_start = pulp.lpSum([y[i][t] for i in player_ids if positions[i] == 3])
        prob += mids_start >= 2, f"MIDs_Start_Min_{t}"
        prob += mids_start <= 5, f"MIDs_Start_Max_{t}"
        
        fwds_start = pulp.lpSum([y[i][t] for i in player_ids if positions[i] == 4])
        prob += fwds_start >= 1, f"FWDs_Start_Min_{t}"
        prob += fwds_start <= 3, f"FWDs_Start_Max_{t}"
        
        # Max 3 players per physical team
        for team in unique_teams:
            prob += pulp.lpSum([s[i][t] for i in player_ids if teams[i] == team]) <= 3, f"Team_{team}_Max_{t}"
            
    # 4. Objective Func: Expected Points (Starter + Captain bonus) + Auto-Sub Expectation, MINUS hits
    # The starter gets full expected points based on probability
    total_starter_xp = pulp.lpSum([xp[i][t] * probs[i] * (y[i][t] + c[i][t]) for i in player_ids for t in T])
    
    # Auto-Sub assumption: If starter misses (1 - probs), the bench collectively replaces them with their own prob
    benched_vars = {i: {t: s[i][t] - y[i][t] for t in T} for i in player_ids}
    total_bench_xp = pulp.lpSum([xp[i][t] * probs[i] * benched_vars[i][t] * 0.15 for i in player_ids for t in T]) # Approx. 15% aggregate chance of autosub occurrence
    
    total_hit_deductions = pulp.lpSum([4.0 * hits[t] for t in T])
    
    prob += total_starter_xp + total_bench_xp - total_hit_deductions, "Objective"
        
    # We will store the top N solutions here
    solutions = []
    
    for solution_idx in range(top_n):
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[status] != 'Optimal':
            break
            
        # Parse output timeline
        timeline = [] 
        
        selected_vars_gw0 = []
        for t in T:
            squad_t = []
            bought_t = []
            sold_t = []
            for i in player_ids:
                if round(pulp.value(s[i][t])) == 1:
                    player_data = df[df['id'] == i].copy()
                    player_data['is_starter'] = (round(pulp.value(y[i][t])) == 1)
                    player_data['is_captain'] = (round(pulp.value(c[i][t])) == 1)
                    # Expose the pure decomposed/probability adjusted xP per player
                    player_data[gw_list[t]] = round(xp[i][t], 2)
                    squad_t.append(player_data)
                    if t == 0:
                        selected_vars_gw0.append(s[i][t])
                if round(pulp.value(b[i][t])) == 1:
                    player_data = df[df['id'] == i].copy()
                    player_data[gw_list[t]] = round(xp[i][t], 2)
                    bought_t.append(player_data)
                if round(pulp.value(sell[i][t])) == 1:
                    sold_t.append(df[df['id'] == i])
                    
            timeline.append({
                'gw': gw_list[t],
                'squad': pd.concat(squad_t, ignore_index=True) if squad_t else pd.DataFrame(),
                'bought': pd.concat(bought_t, ignore_index=True) if bought_t else pd.DataFrame(),
                'sold': pd.concat(sold_t, ignore_index=True) if sold_t else pd.DataFrame(),
                'hits': round(pulp.value(hits[t])),
                'ft_avail': round(pulp.value(ft_avail[t])),
                'ft_carried': round(pulp.value(ft_carried[t])),
                'bank': pulp.value(bank[t])
            })
            
        net_xp = pulp.value(prob.objective)
        solutions.append((timeline, net_xp))
        
        # Add No-Good cut based on the initial GW0 squad
        # Sum of the previously selected 15 players in GW0 must be <= 14 in the next iteration
        prob += pulp.lpSum(selected_vars_gw0) <= 14, f"No_Good_Cut_{solution_idx}"
        
    return solutions

def print_squad_by_gameweeks(timeline: List[Dict]):
    """Prints the selected squad cleanly grouped by gameweek, along with its specific transfers."""
    pos_order = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    
    for state in timeline:
        gw = state['gw']
        squad_df = state['squad']
        squad_df['pos_order'] = squad_df['position'].map(pos_order)
        squad_df = squad_df.sort_values(by=['pos_order', gw], ascending=[True, False])
        
        print(f"\n[ --- {gw} --- ]")
        
        if state['bought'].empty:
            print(f"  TRANSFERS: None (Rolling Free Transfer -> Carried: {state['ft_carried']} | Bank: £{state['bank']:.1f}m)")
        else:
            print(f"  TRANSFERS (FT Avail: {state['ft_avail']} | Hits: {state['hits']} | Bank: £{state['bank']:.1f}m):")
            for _, row in state['sold'].iterrows():
                print(f"    OUT: {str(row['web_name']).encode('ascii', 'ignore').decode('ascii'):<20} | Pos: {row['position']}")
            for _, row in state['bought'].iterrows():
                print(f"    IN : {str(row['web_name']).encode('ascii', 'ignore').decode('ascii'):<20} | Pos: {row['position']} | {gw} Proj xP: {row.get(gw, 0.0):.2f}")
                
        # Split into Starters and Bench
        starters_df = squad_df[squad_df['is_starter']].copy()
        bench_df = squad_df[~squad_df['is_starter']].copy()
        
        print(f"\n  STARTING XI")
        print(f"  {'Position':<10} {'Player Name':<25} {'Team ID':<8} {'Cost (£m)':<12} {f'Proj xP':<8}")
        print("  " + "-" * 65)
        
        gw_total_xP = 0
        for _, row in starters_df.iterrows():
            name = str(row['web_name']).encode('ascii', 'ignore').decode('ascii')
            if row['is_captain']:
                name += " (C)"
            
            # Using the exact heavily decomposed xP probability modifier evaluated in objectve
            gw_xP = row.get(gw, 0.0)
            if row['is_captain']:
                gw_xP *= 2.0
                
            gw_total_xP += gw_xP
            print(f"  {row['position']:<10} {name:<25} {row['team']:<8} £{row['now_cost']:<11.1f} {gw_xP:.2f}")
            
        print("  " + "-" * 65)
        print(f"  {'':<45} GW TOTAL: {gw_total_xP:.2f}")
        
        print(f"\n  BENCH")
        for _, row in bench_df.iterrows():
            name = str(row['web_name']).encode('ascii', 'ignore').decode('ascii')
            gw_xP = row.get(gw, 0.0)
            print(f"  {row['position']:<10} {name:<25} {row['team']:<8} £{row['now_cost']:<11.1f} {gw_xP:.2f}")

def main():
    # ==========================================
    # --- USER CONFIGURATION ---
    # Put your FPL Team ID here (e.g. 1234567)
    USER_TEAM_ID = 1  
    
    # Overwrite budget (£m) if you want, or leave as None to fetch your actual bank from the API
    USER_BUDGET = None 
    
    # How many free transfers you want the optimizer to use
    USER_TRANSFERS = 1  
    # ==========================================

    print("Starting FPL Transfer Recommender (5-GW Horizon)...")
    
    api_data = fetch_fpl_data()
    next_gw = get_next_gameweek(api_data)
    if next_gw:
        print(f"\n[INFO] Planning horizon starts at: {next_gw['name']} (GW {next_gw['id']})")
    
    df_players = extract_players_data(api_data)
    
    # 2. Get top players for prediction to save time
    df_players = df_players.sort_values(by='selected_by_percent', ascending=False)
    top_players = df_players['id'].tolist()[:300] 
    my_squad, bank_budget, buy_prices = fetch_team_from_id(USER_TEAM_ID, get_current_gameweek(api_data))
    
    df_players_static = extract_players_data(api_data)
    hist_df, fixt_df = fetch_player_history(df_players_static['id'].tolist(), next_n_fixtures=5)
    
    hist_df = feature_engineering_historical(hist_df, api_data)
    
    train_df = hist_df.copy()
    models = train_lgbm_multi_models(train_df)
    
    df_players_predicted, next_gw_name, gw_list = predict_5gw(models, df_players_static, hist_df, fixt_df, api_data)
    
    if not gw_list:
        print("No upcoming fixtures found.")
        return
        
    print(f"\nOptimization Horizon: {gw_list}")
    
    solutions = optimize_transfers(
        df=df_players_predicted, 
        current_team_ids=my_squad, 
        bank_budget=bank_budget, 
        initial_transfers=USER_TRANSFERS,
        gw_list=gw_list,
        top_n=1,
        buy_prices=buy_prices
    )
    
    if solutions:
        for i, (timeline, net_xp) in enumerate(solutions, 1):
            print("\n\n" + "="*80)
            print(f"  O P T I O N   {i}   (NET 5GW OBJECTIVE SCORE: {net_xp:.2f})")
            print("="*80)
            print_squad_by_gameweeks(timeline)

if __name__ == "__main__":
    main()
