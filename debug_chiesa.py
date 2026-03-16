import requests
import pandas as pd

url_base = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url_base).json()

jugadores = pd.DataFrame(data['elements'])
jugadores['chance_of_playing_next_round'] = jugadores['chance_of_playing_next_round'].fillna(100)
equipos = pd.DataFrame(data['teams'])
posiciones = pd.DataFrame(data['element_types'])
                        

url_fixtures = "https://fantasy.premierleague.com/api/fixtures/"
fixtures_df = pd.DataFrame(requests.get(url_fixtures).json())

def calculate_xP_by_gw(player_row, gw_number, fixtures_df):
    team_id = player_row['team']
    player_row = player_row.fillna(0)
    position = player_row['element_type']

    fixtures = fixtures_df[
        ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id))
        & (fixtures_df['event'] == gw_number)
    ]

    if fixtures.empty:
        return 0

    chance = player_row.get("chance_of_playing_next_round", 100)
    if pd.isna(chance): chance = 100
    proba_jugar = chance / 100.0

    minutes = max(player_row['minutes'], 1)
    starts = max(player_row['starts'], 1)
    appearances = max(player_row.get('appearances', starts), 1)

    expected_minutes = minutes / appearances
    fraccion_partido = expected_minutes / 90.0

    MINUTOS_FANTASMA = 180
    minutos_suavizados = minutes + MINUTOS_FANTASMA

    xg = float(player_row.get('expected_goals', player_row['goals_scored']))
    xa = float(player_row.get('expected_assists', player_row['assists']))
    
    xG_per90 = (xg / minutos_suavizados) * 90
    xA_per90 = (xa / minutos_suavizados) * 90

    total_xP_gw = 0
    
    # print debug
    print(f"Name: {player_row['web_name']}")
    print(f"Minutes: {minutes}, Starts: {starts}, Appearances: {appearances}")
    print(f"Expected mins: {expected_minutes}, Fraccion: {fraccion_partido}")
    print(f"xg: {xg}, xa: {xa}")
    print(f"xG_per90: {xG_per90}, xA_per90: {xA_per90}")

    for _, fixture in fixtures.iterrows():
        is_home = fixture['team_h'] == team_id
        difficulty = fixture['team_h_difficulty'] if is_home else fixture['team_a_difficulty']
        
        FDR = 1.5 - (difficulty - 1) * (1.0 / 4)
        home_adv = 1.05 if is_home else 0.95

        goal_points = {1: 6, 2: 6, 3: 5, 4: 4}[position]
        xG_points = xG_per90 * goal_points * FDR * home_adv * fraccion_partido
        
        xA_points = xA_per90 * 3 * FDR * home_adv * fraccion_partido
        
        clean_sheet_points = {1: 4, 2: 4, 3: 1, 4: 0}[position]
        cs_prob_base = player_row['clean_sheets'] / starts
        cs_prob = min(cs_prob_base, 0.6)
        
        if expected_minutes < 60:
            cs_prob = 0 
            
        xCS = cs_prob * clean_sheet_points * FDR * home_adv
        
        xSaves = 0
        if position == 1:
            saves = player_row['saves']
            FDR_saves_map = {1: 0.7, 2: 0.85, 3: 1.0, 4: 1.15, 5: 1.3}
            xSaves = (saves / minutos_suavizados) * 90 * 0.33 * FDR_saves_map.get(difficulty, 1.0) * fraccion_partido

        goles_en_contra_pts = 0
        if position in [1, 2]:
            gc_per90 = (player_row['goals_conceded'] / minutos_suavizados) * 90
            goles_en_contra_pts = (gc_per90 / 2) * -1 * (2 - FDR) * fraccion_partido

        minute_points = 0
        prob_titular = starts / appearances

        if expected_minutes >= 60:
            minute_points = 2 * prob_titular + 1 * (1 - prob_titular)
        elif expected_minutes > 0:
            minute_points = 1 * prob_titular
            
        ict_index = float(player_row.get("ict_index", 0)) / 10
        bonus_points = ((xG_points + xA_points + xCS) * 0.2) + (ict_index * 0.05 * FDR * fraccion_partido)
        
        penalty_points = 0
        penalty_points -= 1 * (player_row.get('yellow_cards', 0) / minutos_suavizados * 90) * fraccion_partido
        penalty_points -= 3 * (player_row.get('red_cards', 0) / minutos_suavizados * 90) * fraccion_partido
        penalty_points -= 2 * (player_row.get('penalties_missed', 0) / minutos_suavizados * 90) * fraccion_partido

        partido_xP = (
            xG_points + xA_points + xCS + xSaves +
            bonus_points + minute_points + penalty_points + goles_en_contra_pts
        )
        print(f"Partido xP pre-proba: {partido_xP}")
        print(f"Details: xG_pts: {xG_points}, xA_pts: {xA_points}, xCS: {xCS}, min_pts: {minute_points}, bonus: {bonus_points}, penalty: {penalty_points}, gc: {goles_en_contra_pts}")
        total_xP_gw += partido_xP

    final_xP = total_xP_gw * proba_jugar
    print(f"Final xP: {final_xP}")
    return round(final_xP, 2)

chiesa = jugadores[jugadores['web_name'] == 'Chiesa'].iloc[0]
calculate_xP_by_gw(chiesa, 28, fixtures_df)  # Assuming GW is 28

# Let's also check Salah or Palmer for comparison
salah = jugadores[jugadores['web_name'] == 'Salah'].iloc[0]
calculate_xP_by_gw(salah, 28, fixtures_df)
