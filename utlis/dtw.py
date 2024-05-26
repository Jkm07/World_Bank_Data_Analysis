import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pyts.metrics import dtw
import numpy as np

def get_indicator_dtw_matrix(df_indicator):
    indicator_compare_dataframe = pd.DataFrame(index = df_indicator['Country Name'].unique())
        
    for country_main in df_indicator['Country Name'].unique():
        country_compare_series = pd.Series(index = df_indicator['Country Name'].unique())
        country_select = df_indicator['Country Name'] == country_main
        x = df_indicator.loc[country_select, '1991':'2021'].iloc[0]
        for country_compare in df_indicator['Country Name'].unique():
            country_select = df_indicator['Country Name'] == country_compare
            y = df_indicator.loc[country_select, '1991':'2021'].iloc[0]
            country_compare_series[country_compare] = dtw(x, y, method='sakoechiba', options={'window_size': 0.5})
        indicator_compare_dataframe[country_main] = country_compare_series
    return indicator_compare_dataframe

def plot_time_series_compare_matrix(cm, class_names, title, size = 8):
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.show()

def get_sum_of_dtw_matrix(df, category_name):
    print('\033[1m' + category_name + '\033[0m')
    print()
    category_filter = df['Category Name'] == category_name
    array_size = df['Country Name'].nunique()
    sum_of_indicators = np.zeros((array_size, array_size))
    for indicator_name in df[category_filter]['Indicator Name'].unique():
        print(indicator_name)
        scalar = StandardScaler()
        indicator_compare_dataframe = get_indicator_dtw_matrix(df[df['Indicator Name'] == indicator_name])
        sum_of_indicators += scalar.fit_transform(indicator_compare_dataframe)

    columns = df['Country Name'].unique()
    plot_time_series_compare_matrix(sum_of_indicators, columns, category_name, size = 30)
    return pd.DataFrame(sum_of_indicators, columns = columns, index = columns)