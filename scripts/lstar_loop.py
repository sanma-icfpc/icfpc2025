# -*- coding: utf-8 -*-
import os
import requests
import sys

import lstar

def solve(problem_name: str):
    print(f"{problem_name=}")
    # Get ID from env or register once (keep the returned id secret!)
    team_id = os.getenv("ICFP_ID")
    client = lstar.AedificiumClient(team_id=team_id)

    if client.id is None:
        # One-time registration (then set ICFP_ID env for later runs)
        # !!! Replace with your team's information
        # client.register(name="Your Team", pl="Python", email="you@example.com")
        # print("Registered. Your secret id:", client.id)
        sys.exit(1)

    # Select a problem (try the tiny 'probatio' first)
    chosen = client.select_problem(problem_name)
    print("Selected:", chosen)

    oracle = lstar.ExploreOracle(client)
    learner = lstar.LStarMooreLearner(oracle)

    hyp = learner.learn()
    guess_map = lstar.build_guess_from_hypothesis(hyp)

    # Optionally pretty print the guess before submitting
    # print(json.dumps(guess_map, indent=2))

    ok = client.guess(guess_map)
    print("Guess correct?", ok)
    print(f"{oracle.last_query_count=}")
    print(flush=True)


if __name__ == "__main__":
    problem_names = ["probatio", "primus", "secundus", "tertius", "quartus", "quintus"]
    while True:
        for problem_name in problem_names:
            # For safety in an example script; remove in production
            try:
                solve(problem_name)
            except requests.HTTPError as e:
                print("HTTP error:", e)
            except Exception as ex:
                print("Error:", ex)
