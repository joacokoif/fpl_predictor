import pandas as pd
import requests

def debug_api():
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    data = r.json()
    teams = pd.DataFrame(data['teams'])
    print(teams[['id', 'strength', 'strength_attack_home', 'strength_defence_home']].head())
    
debug_api()
