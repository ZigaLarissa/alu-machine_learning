#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns
the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    returns the list of ships that can hold a given number of passengers
    If the list is empty, the method should return an empty list
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            passengers = ship['passengers']
            if passengers.isnumeric():
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
        url = data['next']
    return ships
