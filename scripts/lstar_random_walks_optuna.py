# -*- coding: utf-8 -*-
import os
import requests
import sys
import optuna
import threading

import lstar
import lstar_random_walks


def solve(
    problem_name: str,
    max_loops: int = 200,
    num_random_walks: int = 4,
    max_random_walk_length: int = 16,
):
    print(flush=True)
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
    learner = lstar_random_walks.LStarMooreRandomWalkLearner(
        oracle,
        max_loops,
        num_random_walks,
        max_random_walk_length,
    )

    hyp = learner.learn()
    guess_map = lstar.build_guess_from_hypothesis(hyp)

    # Optionally pretty print the guess before submitting
    # print(json.dumps(guess_map, indent=2))

    ok = client.guess(guess_map)
    print("Guess correct?", ok)
    return ok, oracle.last_query_count


if __name__ == "__main__":
    problem_names = ["probatio", "primus", "secundus", "tertius", "quartus", "quintus"]
    threads = list()
    lock = threading.Lock()
    for problem_name in problem_names:

        def solve_local(problem_name: str, lock: threading.Lock):
            def objective(trial: optuna.Trial):
                with lock:
                    # For safety in an example script; remove in production
                    try:
                        num_random_walks = trial.suggest_int("num_random_walks", 1, 8)
                        max_random_walk_length = trial.suggest_int(
                            "max_random_walk_length", 1, 32
                        )
                        ok, last_query_count = solve(
                            problem_name,
                            num_random_walks=num_random_walks,
                            max_random_walk_length=max_random_walk_length,
                        )
                        energy = last_query_count
                        if not ok:
                            energy += 10000
                        return energy
                    except requests.HTTPError as e:
                        print("HTTP error:", e)
                        return 10000
                    except Exception as ex:
                        print("Error:", ex)
                        return 10000

            study_name = problem_name
            storage = f"sqlite:///random_walk.{study_name}.db"  # SQLite ファイルを指定
            study = optuna.create_study(
                study_name=study_name, storage=storage, load_if_exists=True
            )
            study.optimize(objective, n_trials=sys.maxsize)

        thread = threading.Thread(target=solve_local, args=(problem_name, lock))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
