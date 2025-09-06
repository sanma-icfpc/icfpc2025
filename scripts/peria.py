import requests
import json
import os
import sys
import random
import itertools

SERVER = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com/"
ID = os.getenv("ICFP_ID")

PROBLEM_SET = ["probatio", "primus",
               "secundus", "tertius", "quartus", "quintus"]


def request(command, data):
    body = json.dumps(data)

    print("JSON: " + body, file=sys.stderr)
    url = SERVER + "/" + command
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, data=body, headers=headers)
    print(response.status_code, file=sys.stderr)
    print(response.text, file=sys.stderr)
    return json.loads(response.text)


def select_problem(problem_id):
    problem = PROBLEM_SET[problem_id]
    select_body = {
        "id": ID,
        "problemName": problem,
    }
    request("select", select_body)


def solve(num_rooms):
    plan = ''
    for _ in range(num_rooms * 18):
        plan += str(random.randint(0, 5))
    explore_body = {
        "id": ID,
        "plans": [plan]
    }
    tour = request("explore", explore_body)["results"][0]
    print(tour, file=sys.stderr)
    print(num_rooms, file=sys.stderr)

    tags = []
    doors = [[None, None, None, None, None, None] for _ in range(num_rooms)]
    current = tour[0]
    for (door, tag) in zip(plan, tour[1:]):
        door = int(door)
        if doors[current][door] is None:
            doors[current][door] = tag
        else:
            assert doors[current][door] == tag, 'Incosistent of room {} - door{}: {}, {}'.format(
                current, door, doors[current][door], tag)
        current = tag
        if tag not in tags:
            tags.append(tag)
    print(doors, file=sys.stderr)

    # Connect known doors
    targets = [[[] for _ in range(num_rooms+1)] for _ in range(num_rooms)]
    for rid in range(num_rooms):
        for did in range(6):
            target = doors[rid][did]
            target = -1 if target is None else target
            targets[rid][target].append(did)
    print(targets, file=sys.stderr)
    for r0 in range(num_rooms):
        # Self-room loop doors.
        for d in targets[r0][r0]:
            doors[r0][d] = [r0, d]
        targets[r0][r0] = []
        # Connect doors between known rooms.
        for r1 in range(r0 + 1, num_rooms):
            while len(targets[r0][r1]) > 0 and len(targets[r1][r0]) > 0:
                d0 = targets[r0][r1].pop(-1)
                d1 = targets[r1][r0].pop(-1)
                doors[r0][d0] = [r1, d1]
                doors[r1][d1] = [r0, d0]

    print(doors, file=sys.stderr)
    print(targets, file=sys.stderr)

    # Connect rooms using unknown doors.
    for r0 in range(num_rooms):
        for r1 in range(num_rooms):
            while len(targets[r0][r1]) > 0:
                assert len(targets[r1][num_rooms]) > 0
                d0 = targets[r0][r1].pop(-1)
                d1 = targets[r1][num_rooms].pop(-1)
                doors[r0][d0] = [r1, d1]
                doors[r1][d1] = [r0, d0]

    print(doors, file=sys.stderr)
    print(targets, file=sys.stderr)

    # Still we have unknown doors, assume them self-connected.
    for r in range(num_rooms):
        for d in targets[r][num_rooms]:
            doors[r][d] = [r, d]

    print(doors, file=sys.stderr)

    # Build answers
    rooms = [i % 4 for i in range(num_rooms)]
    starting_room = tour[0]
    connections = []
    for r0 in range(num_rooms):
        for d0 in range(6):
            (r1, d1) = doors[r0][d0]
            if r1 < r0 or (r0 == r1 and d1 <= d0):
                connection = {
                    "from": {"room": r0, "door": d0},
                    "to": {"room": r1, "door": d1},
                }
                connections.append(connection)
    guess_body = {
        "id": ID,
        "map": {
            "rooms": rooms,
            "startingRoom": starting_room,
            "connections": connections,
        }
    }
    request("guess", guess_body)


def main():
    problem_id = 0
    select_problem(problem_id)
    num_rooms = 3 if problem_id == 0 else (problem_id * 6)
    solve(num_rooms)


if __name__ == "__main__":
    main()
