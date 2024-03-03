import pandas as pd
import os

def get_dataset_indicator(file_name: str, indicator_name: str):
    data = pd.read_csv(file_name, header = 2)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data[(data['Indicator Name'] == indicator_name)]
    data = data.drop(columns=['Indicator Name', 'Indicator Code'])
    data = data.set_index('Country Code')
    return data

def get_dataset_country(file_name: str, country_name: str):
    data = pd.read_csv(file_name, header = 2)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data[(data['Country Name'] == country_name)]
    data = data.drop(columns=['Country Name', 'Country Code'])
    data = data.set_index('Indicator Code')
    return data

def get_dataset_all_indicators(file_name: str):
    data = pd.read_csv(file_name, header = 2)
    indicators_series = data['Indicator Name'].unique()
    result = {}
    for indicator_name in indicators_series:
        result[indicator_name] = get_dataset_indicator(file_name, indicator_name)
    return result

def get_dataset_all_countries(file_name: str):
    data = pd.read_csv(file_name, header = 2)
    country_series = data['Country Name'].unique()
    result = {}
    for country_name in country_series:
        result[country_name] = get_dataset_indicator(file_name, country_name)
    return result

def get_dataset_all_category_indicators() -> dict:
    file_names = os.listdir('data')
    result = {}
    for file_name in file_names:
        result[file_name[:-4]] = get_dataset_all_indicators(f"data/{file_name}")
    return result

def concat_all_datasets(datasets: dict = None) -> pd.DataFrame:
    if datasets == None:
        datasets = get_dataset_all_category_indicators()
    datasests_list = []
    for category_name in datasets:
        for indicator_name in datasets[category_name]:
            curr_dataframe = datasets[category_name][indicator_name].copy()
            curr_dataframe.insert(0, 'Category Name', category_name)
            curr_dataframe.insert(0, 'Indicator Name', indicator_name)
            curr_dataframe = curr_dataframe.set_index(['Category Name', 'Indicator Name'], append=True)
            datasests_list.append(curr_dataframe)
    return pd.concat(datasests_list)
    