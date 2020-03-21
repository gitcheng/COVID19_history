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

def plot_cumulated_histories(ax, df, i0=0, i1=10, title=None, yscale='log', ymax=None, case='confirmed cases', plot_remaining=False):
    '''
    Plot the history of countries that have the i0-th to i10-th highest confirmed case
    
    ax: axes
    case: 'confirmed cases' or 'deaths'
    '''
    dfp = df.sort_values(by=df.columns[-1], ascending=False)
    for row in dfp[i0:i1].iterrows():
        label = '{} ({})'.format(row[0], int(row[1][-1]))
        ax.plot_date(row[1].index, row[1], fmt='-', label=label, lw=3, alpha=0.8);
        
    if plot_remaining:
        row = dfp[i1:].sum()
        label = 'Others ({})'.format(int(row[-1]))
        ax.plot_date(row.index, row, fmt='-', label=label, lw=5, alpha=0.2, color='gray')
        
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)
    if ymax is None:
        ymax = 2 * dfp[df.columns[-1]].max()
        
    ax.set_ylim(1, ymax)
    ax.set_yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    if case == 'deaths':
        legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# deaths)')
    else:
        legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# cases)')
    plt.setp(legend.get_title(),fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Reported {}'.format(case), fontsize='xx-large');
    if title is not None:
        ax.set_title('{} ({})'.format(title, df.columns[-1]), fontsize='xx-large')
    
def plot_average_rate(ax, df0, ndays, countries, threshold=10):
    '''
    Plot history of n-day average of daily increase percentage of selected countries.
    
    df: DataFrame
    ndays: int
    countries: a list of strings (country names)
    '''
    df = df0.loc[countries]

    for idx, row in df.iterrows():
        selrow = row[row >= threshold]
        diff = selrow.diff(ndays)
        averate = (selrow / (selrow - diff))**(1/ndays) - 1
        label = '{} ({}%)'.format(idx, int(100*averate[-1]))
        ax.plot_date(selrow.index, 100*averate, fmt='-', label=label, lw=3, alpha=0.7);
        
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)
        
    ax.set_ylim(-1,101)
    #plt.yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (rate)')
    plt.setp(legend.get_title(), fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Rate (%)', fontsize='xx-large');
    ax.set_title('Confirmed case daily increase rate ({}-day average) of selected countries ({})'.format(ndays, df.columns[-1]), fontsize='xx-large')


def plot_cumulated_since(ax, df0, countries, threshold=100, yscale='log'):
    '''
    Plot cumulated cases since the number of cases is confirmed
    
    df: DataFrame
    countries: a list of strings (country names)
    '''
    df = df0.loc[countries]

    for j, (idx, row) in enumerate(df.iterrows()):
        selrow = row[row >= threshold]
        label = '{} ({})'.format(idx, int(selrow[-1]))
        if j < 10:
            ax.plot(selrow.values, 'C{}-'.format(j), label=label, lw=3, alpha=0.7);
            ax.scatter(len(selrow)-1, selrow.values[-1], s=50, marker='o', color='C{}'.format(j%10), linewidths=2)
        else:
            ax.plot(selrow.values, 'C{}--'.format(j%10), label=label, lw=3, alpha=0.7);
            ax.scatter(len(selrow)-1, selrow.values[-1], s=50, marker='o', color='C{}'.format(j%10), facecolor='none', linewidths=2)
        
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    ax.set_yscale(yscale)  
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (# cases)')
    plt.setp(legend.get_title(),fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Reported confirmed cases', fontsize='xx-large');
    ax.set_xlabel('Days since {}-th case'.format(threshold), fontsize='xx-large')
    ax.set_title('Confirmed case of selected countries days since {}-th case ({})'.format(threshold, df.columns[-1]), fontsize='xx-large')
