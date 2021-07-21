import os
import pandas as pd
import numpy as np

path_to_here = os.path.dirname(os.path.realpath(__file__))

file_csv = f'{path_to_here}/MI_traits_EDTable1.csv'

df = pd.read_csv(file_csv, na_values='NaN')

_str_columns = ['Species', 'Reference']

def _convert(s):
    """convert column to numeric type"""
    try:
        if s == 'NaN':
            return np.nan
        elif '(' in s:
            return np.float(s.split(' ')[0])
        else:
            return np.float(s)
    except:
        print(f'could not convert string:\n{s}')
        raise

_converter = {k: _convert for k in df.columns if k not in _str_columns}
_converter.update({
    k: lambda s: str(s.replace("'", '')).strip() 
    for k in _str_columns
})

df_full = pd.read_csv(file_csv, 
                      na_values='NaN', 
                      converters=_converter
                     )

df_mi = pd.DataFrame(dict(
    Species=df_full['Species'],
    Ao=1. /d f_full['Vh (atm)'],
    Ac=1. / df_full['Vh (atm)'] / df_full['Phi_crit (histogram)'],
    Eo=df_full['Eo (eV)'],
))
df_mi = df_mi.dropna()

df_mi.Ao.attrs = dict(
    long_name='Hypoxic tolerance',
    units='atm',
)
df_mi.Ac.attrs = dict(
    long_name='Hypoxic tolerance (normalized to Φcrit)',
    units='atm',
)
df_mi.Eo.attrs = dict(
    long_name='Temperature sensitivity',
    units='eV',
)


