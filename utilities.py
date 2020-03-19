import pandas as pd
import matplotlib.pyplot as plt

def merge_locals(df):
    # merge rows of the same country
    country_names = df['Country/Region'].unique()
    columns = df.columns
    df_c = pd.DataFrame(columns=columns[4:])
    for cn in country_names:
        row = df[df['Country/Region']==cn].sum()
        row = row[4:]
        if cn.endswith('*'):
            row.name = cn[:-1]
        else:
            row.name = cn
        df_c = df_c.append(row)
    #df_c = df_c.sort_values(by=columns[-1], ascending=False)
    return df_c

def plot_cumulated_histories(df, i0=0, i1=10, title=None, figsize=(15,6), yscale='log', ymax=None, case='confirmed cases', plot_remaining=False):
    '''
    Plot the history of countries that have the i0-th to i10-th highest confirmed case
    
    case: 'confirmed cases' or 'deaths'
    '''
    dfp = df.sort_values(by=df.columns[-1], ascending=False)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111) 
    for row in dfp[i0:i1].iterrows():
        label = '{} ({})'.format(row[0], int(row[1][-1]))
        plt.plot_date(row[1].index, row[1], fmt='-', label=label, lw=3, alpha=0.8);
        
    if plot_remaining:
        row = dfp[i1:].sum()
        label = 'Others ({})'.format(int(row[-1]))
        plt.plot_date(row.index, row, fmt='-', label=label, lw=5, alpha=0.2, color='gray')
        
    plt.xticks(fontsize='large', rotation=45)
    plt.yticks(fontsize='large')
    if ymax is None:
        ymax = 2 * dfp[df.columns[-1]].max()
        
    plt.ylim(1, ymax)
    plt.yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    if case == 'deaths':
        legend = plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# deaths)')
    else:
        legend = plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# cases)')
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.grid(axis='y');
    plt.ylabel('Reported {}'.format(case), fontsize='x-large');
    if title is not None:
        plt.title('{} ({})'.format(title, df.columns[-1]), fontsize='xx-large')
    
def plot_average_rate(df0, ndays, countries, threshold=10, figsize=(15,6)):
    '''
    Plot history of n-day average of daily increase percentage of selected countries.
    
    df: DataFrame
    ndays: int
    countries: a list of strings (country names)
    '''
    df = df0.loc[countries]

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111) 
    for idx, row in df.iterrows():
        label = '{} ({})'.format(idx, int(row[-1]))
        selrow = row[row >= threshold]
        diff = selrow.diff(ndays)
        averate = (selrow / (selrow - diff))**(1/ndays) - 1
        plt.plot_date(selrow.index, 100*averate, fmt='-', label=label, lw=3, alpha=0.7);
        
    plt.xticks(fontsize='large', rotation=45)
    plt.yticks(fontsize='large')
        
    plt.ylim(-1,101)
    #plt.yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# cases)')
    plt.setp(legend.get_title(),fontsize='x-large')
    plt.grid(axis='y');
    plt.ylabel('Rate (%)', fontsize='xx-large');
    plt.title('Confirmed case daily increase rate ({}-day average) of selected countries ({})'.format(ndays, df.columns[-1]), fontsize='xx-large')



