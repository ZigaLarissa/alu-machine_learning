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
    sentient_planets = []
    sentient_classifications = ["sentient"]

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data["results"]:
            if (
                species["classification"].lower() in sentient_classifications
                and species["homeworld"]
            ):
                homeworld_url = species["homeworld"]
                homeworld_response = requests.get(homeworld_url)
                homeworld_data = homeworld_response.json()
                sentient_planets.append(homeworld_data["name"])

        url = data["next"]

    return sentient_planets
