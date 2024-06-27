#!/usr/bin/env python3
"""
By using the Swapi API, create a method that returns
the list of names of the home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home
    planets of all sentient species.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = []
    while url:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            if species["classification"] == "sentient":
                if species["homeworld"]:
                    response = requests.get(species["homeworld"])
                    planet = response.json()
                    planets.append(planet["name"])
        url = data["next"]
    return planets
