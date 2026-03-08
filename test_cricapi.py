import json, requests

api_key = "c8553d8b-1b30-41d2-b6d1-145234c70dee"
# Get matches
r = requests.get(f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}")
if r.status_code == 200:
    with open("cricapi_matches.json", "w") as f:
        json.dump(r.json(), f, indent=2)
        
    matches = r.json().get("data", [])
    ind_nz_match = next((m for m in matches if "India" in m.get("name", "") and "New Zealand" in m.get("name", "")), None)
    
    if ind_nz_match:
        mid = ind_nz_match["id"]
        # get score or match info
        r_score = requests.get(f"https://api.cricapi.com/v1/match_info?apikey={api_key}&offset=0&id={mid}")
        if r_score.status_code == 200:
            with open("cricapi_match.json", "w") as f:
                json.dump(r_score.json(), f, indent=2)
