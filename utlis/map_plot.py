import geopandas as gpd

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