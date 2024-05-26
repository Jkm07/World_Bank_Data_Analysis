import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def map_values_index(country_series):
    return country_series.rename(index={
        'Central African Republic': 'South Africa',
        'Egypt, Arab Rep.' : 'Egypt',
        'Gambia, The' : 'Gambia',
        'Korea, Rep.' : 'South Korea',
        'Russian Federation' : 'Russia',
        'Slovak Republic' : 'Slovakia',
        'Turkiye' : 'Turkey',
        'United States' : 'United States of America'
          })
def map_values_columns(countries, columns_name):
    return countries[columns_name].replace({
        'Central African Republic': 'South Africa',
        'Egypt, Arab Rep.' : 'Egypt',
        'Gambia, The' : 'Gambia',
        'Korea, Rep.' : 'South Korea',
        'Russian Federation' : 'Russia',
        'Slovak Republic' : 'Slovakia',
        'Turkiye' : 'Turkey',
        'United States' : 'United States of America'
    })

def plot_map(clusters, title):

    clusters = clusters.copy()

    clusters += 1
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    clusters = clusters.reset_index()

    clusters['Country Name'] = map_values_columns(clusters, 'Country Name')

    world_clusters = world.merge(clusters, how='left', left_on='name', right_on='Country Name')

    fig, ax = plt.subplots(figsize=(20, 20))

    world.plot(ax=ax, color='lightgrey', edgecolor='black')

    world_clusters.plot(column='Cluster', ax=ax, legend=True, categorical=True,)

    ax.set_title(title)

