import requests
import json

SERVER = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com/"
ID = "sanma-icfpc@googlegroups.com BLVs94q7eF92v8udVT6Egw"


def main():
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "id": ID,
        "problemName": "probatio",
    }
    json_str = json.dumps(data)

    print("JSON: " + json_str)
    url = SERVER + "/select"
    response = requests.post(url, data=json_str, headers=headers)
    print(response.status_code)
    print(response.text)


if __name__ == "__main__":
    main()
