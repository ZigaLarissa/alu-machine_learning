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
    
    while species_url:
        response = requests.get(species_url)
        data = response.json()
        
        for species in data['results']:
            if species['designation'] == 'sentient' or species['classification'] == 'sentient':
                homeworld_url = species['homeworld']
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    homeworld_data = homeworld_response.json()
                    planets.append(homeworld_data['name'])
        
        species_url = data['next']  # Move to the next page

    return planets
