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
    url = 'https://swapi-api.alx-tools.com/api/species/'
    home_names = []

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data['results']:
            if (species['classification'] == 'sentient' 
            and species['homeworld'] != 'unknown' 
            and species['homeworld'] != 'null'):
                home_url = (species['homeworld'])
                home_response = requests.get(home_url)
                home_data = home_response.json()
                home_name = home_data['name']
                home_names.append(home_name)

        url = data['next']

    return home_names