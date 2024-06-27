#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns
the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    - The method takes an integer as a parameter
    and returns a list of ships that can hold that many passengers
    - add the pagination
    - If no ship available, return an empty list.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for ship in data["results"]:
            if int(ship["passengers"]) >= passengerCount:
                ships.append(ship["name"])
        url = data["next"]
    return ships
