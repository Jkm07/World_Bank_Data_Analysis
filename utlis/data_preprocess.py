import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fill_missing_with_mean(data):
    """
    Uzupełnia brakujące wartości w danych średnią z pozostałych wartości w danej kolumnie.

    Parametry:
    - data: DataFrame zawierający dane

    Zwraca:
    DataFrame z uzupełnionymi brakującymi wartościami.
    """
    filled_data = data.copy()

    filled_data.loc[:,"1991":"2021"] = filled_data.loc[:,"1991":"2021"].interpolate('linear', axis =1, limit_direction='both')
    return filled_data

def min_max_scaling_by_specific_category(data):
    """
    Przeprowadza min-max scaling danych dla wszystkich kategorii szczegółowych.

    Parametry:
    - data: DataFrame zawierający dane, z kategoriami ogólną i szczegółową w trzeciej i czwartej kolumnie

    Zwraca:
    DataFrame z przeskalowanymi danymi.
    """
    # Tworzymy kopię danych, aby uniknąć zmiany oryginalnych danych
    scaled_df = data.copy()
    
    # Pobieramy unikalne kategorie ogólne
    general_categories = scaled_df['Category Name'].unique()
    
    for general_category in general_categories:
        # Sprawdzamy, jakie kategorie szczegółowe zawierają daną kategorię ogólną
        specific_categories = scaled_df.loc[scaled_df['Category Name'] == general_category, 'Indicator Name'].unique()
        for specific_category in specific_categories:
            # Wybieramy dane dla danej kategorii ogólnej i wybranej kategorii szczegółowej
            selected_data = scaled_df[(scaled_df['Category Name'] == general_category) & (scaled_df['Indicator Name'] == specific_category)]
            # Wybieramy tylko kolumny numeryczne do skalowania
            numeric_cols = selected_data.select_dtypes(include=['float64', 'int64']).columns
            data_to_scale = selected_data[numeric_cols]
            # Przeprowadzamy skalowanie
            scaler = MinMaxScaler()
            if len(data_to_scale.columns) > 0:
                scaled_data = scaler.fit_transform(data_to_scale)
                scaled_df.loc[selected_data.index, numeric_cols] = scaled_data
    
    return scaled_df