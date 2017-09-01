import os
from itertools import product

import click

import numpy as np
import pandas as pd

@click.command()
@click.argument('inputs', nargs=-1, type=click.Path())
def summarize_cross_validation(inputs):
    
    def q10(x):
        return np.percentile(x, q=10)
    def q90(x):
        return np.percentile(x, q=90)
    error_cols = ['10%', '50%', '90%']
    region_cols =['0.5', '0.75', '0.9']
    area_cols = ['State', 'County', 'City']
    
    table_cols = ['Seeds', 'Model'] + error_cols + region_cols + area_cols
    table = pd.DataFrame(columns=table_cols)
    
    for input_ in inputs:
        df = pd.read_csv(input_)
        print(df)
        
        if df['seeds'].unique() == 'none':
            df.set_index(['ids', 'seeds', 'model'], inplace=True)
            
            error_summary = df['error'].groupby(level=['seeds', 'model']).agg([q10, np.median, q90])
            error_summary.columns = error_cols
            
            region_summary = 100 * df[region_cols].astype(float).groupby(level=['seeds', 'model']).mean()
            
            areas = ['over_state', 'over_county', 'over_city']
            area_summary = 100 * df[areas].astype(float).groupby(level=['seeds', 'model']).mean()
            area_summary.columns = area_cols
            
            summary = pd.concat([error_summary, region_summary, area_summary], axis=1)
            summary.index.rename([name.title() for name in summary.index.names], inplace=True)
            summary.reset_index(inplace=True)
            summary['Seeds'] = summary['Seeds'].str.title()
            table = pd.concat([table, summary], axis=0)
        
        else:
            df.set_index(['ids', 'seeds', 'model'], inplace=True)
            
            error_summary = df['error'].groupby(level=['seeds', 'model']).agg([q10, np.median, q90])
            error_summary.columns = error_cols
            
            region_summary = 100 * df[region_cols].astype(float).groupby(level=['seeds', 'model']).mean()
            
            areas = ['over_state', 'over_county', 'over_city']
            area_summary = 100 * df[areas].astype(float).groupby(level=['seeds', 'model']).mean()
            area_summary.columns = area_cols
            
            summary = pd.concat([error_summary, region_summary, area_summary], axis=1)
            seeds = summary.index.get_level_values('seeds').unique().values
            idx_order = list(product(seeds, ['Spatial KNN', 'Spatial RF', 'Spatial NN', 'DeepSpace']))
            summary = summary.loc[idx_order]
            summary.index.rename([name.title() for name in summary.index.names], inplace=True)
            summary.reset_index(inplace=True)
            summary['Seeds'] = summary['Seeds'].str.title()
            table = pd.concat([table, summary], axis=0)
    
    table = table[table_cols]
    print(table.to_latex(index=False, float_format=lambda x: "{0:.1f}".format(x), na_rep='-'))

if __name__ == '__main__':
    summarize_cross_validation()
