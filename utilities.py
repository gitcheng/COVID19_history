import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def merge_locals(df):
    '''
    merge rows of the same country
    '''
    country_names = df['Country/Region'].unique()
    columns = df.columns
    df_c = pd.DataFrame(columns=columns[4:], dtype=int)
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

def us_states_data(df):
    '''
    Retrieve US data by states and territories
    '''
    us_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', \
                 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Idaho', \
                 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',\
                 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',\
                 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', \
                 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island',\
                 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virgin Islands',\
                 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    state_abbr = dict(AL='Alabama', AK='Alaska', AZ='Arizona', AR='Arkansas', CA='California', CO='Colorado',\
                      CT='Connecticut', DE='Delaware', DC='District of Columbia', FL='Florida', GA='Georgia',\
                      HI='Hawaii', ID='Idaho', IL='Illinois', IN='Indiana', IA='Iowa', KS='Kansas', \
                      KY='Kentucky', LA='Louisiana', ME='Maine', MD='Maryland', MA='Massachusetts', \
                      MI='Michigan', MN='Minnesota', MS='Mississippi', MO='Missouri', MT='Montana', \
                      NE='Nebraska', NV='Nevada', NH='New Hampshire', NJ='New Jersey', NM='New Mexico',
                      NY='New York', NC='North Carolina', ND='North Dakota', OH='Ohio', OK='Oklahoma', \
                      OR='Oregon', PA='Pennsylvania', RI='Rhode Island', SC='South Carolina', SD='South Dakota',\
                      TN='Tennessee', TX='Texas', UT='Utah', VT='Vermont', VA='Virginia', WA='Washington',\
                      WV='West Virginia', WI='Wisconsin', WY='Wyoming')
    state_abbr['D.C.'] = state_abbr['DC']
    
    columns = df.columns
    ret = pd.DataFrame(columns=columns[4:], dtype=int)
    for idx, row in df.iterrows():
        if row['Province/State'] in us_states:
            myrow = row[4:]
            myrow.name = row['Province/State']
            ret = ret.append(myrow)
            
    # add county/city data back to state
    for idx, row in df.iterrows():
        state = row['Province/State']
        if type(state) != str:
            continue
        fields = [x.strip() for x in state.split(',')]
        if len(fields) < 2:
            continue
        abb = fields[-1]
        state = state_abbr.get(abb)
        if not state:
            continue
        ret.loc[state] = ret.loc[state] + row[4:]
        
    return ret
    

def region_label(key):
    dd = {'United Kingdom': 'UK', 'Korea, South': 'Korea, S.'}
    ret = dd.get(key)
    if ret is None:
        ret = key
    return ret

    
def plot_cumulated_histories(ax, df, i0=0, i1=10, title=None, yscale='log', ymax=None, case='confirmed cases', plot_remaining=False, starting_date=None):
    '''
    Plot the history of countries that have the i0-th to i10-th highest confirmed case
    
    ax: axes
    df: dataframe
    i0, i1: (i0+1)-th to i1-th highest are ploted.
    case: 'confirmed cases' or 'deaths'
    '''
    dfp = df.sort_values(by=df.columns[-1], ascending=False)
    for idx, row in dfp[i0:i1].iterrows():
        region = region_label(idx)
        label = '{} ({})'.format(region, int(row[-1]))
        ax.plot_date(row.index, row, fmt='-', label=label, lw=3, alpha=0.8);
        
    if plot_remaining:
        row = dfp[i1:].sum()
        label = 'Others ({})'.format(int(row[-1]))
        ax.plot_date(row.index, row, fmt='-', label=label, lw=5, alpha=0.2, color='gray')
        
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)
    if ymax is None:
        if yscale == 'log':
            ymax = 2 * dfp[df.columns[-1]].max()
        else:
            ymax = 1.07 * dfp[df.columns[-1]].max()
    if yscale == 'log':
        ax.set_ylim(0.8, ymax)
    else:
        ax.set_ylim(0, ymax)

    if starting_date is not None:
        ax.set_xlim(pd.to_datetime(starting_date))
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
        ax.set_title('{} (update {})'.format(title, df.columns[-1]), fontsize='xx-large')
    

def plot_fatality_ratio(ax, df_death, df_confirmed, i0=0, i1=10, min_cases=1000, threshold=100, title=None, yscale='linear', legend_title='Country'):
    '''
    Plot the case fatality ratio to date (deaths/confirmed)

    ax: axes
    df_death: dataframe of deaths
    df_confirmed: dataframe of confirmed cases
    i0, i1: (i0+1)-th to i1-th highest are ploted
    min_cases: the minimum number of confirmed cases
    '''
    sel = df_confirmed.iloc[:,-1] >= min_cases

    dfd = df_death.loc[sel]
    dfc = df_confirmed.loc[sel]

    df_ratio = 100 * dfd / dfc
    dfp = df_ratio.sort_values(by=df_ratio.columns[-1], ascending=False)
    for idx, row in dfp[i0:i1].iterrows():
        region = region_label(idx)
        label = '{} ({:.2f}%)'.format(region, row[-1])
        sel = dfc.loc[idx] >= threshold
        ax.plot_date(row.loc[sel].index, row.loc[sel], fmt='-', label=label, lw=3, alpha=0.8);

    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)

    ax.set_yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (ratio)'.format(legend_title))
    plt.setp(legend.get_title(),fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Reported case fatality ratio', fontsize='xx-large');
    if title is not None:
        ax.set_title('{} (update {})'.format(title, df_ratio.columns[-1]), fontsize='xx-large')
    

def doubling_time(rate):
    '''
    Return a string of the length of time to double

    rate: daily increase fraction
    '''
    if rate < 1e-6:
        return 'never'
    days = np.log(2) / np.log(1 + rate)
    if days > 1000:
        return 'long time'
    if days >= 365:
        ret = '{:.1f} y'.format(days/365)
        return ret
    if days >= 30:
        ret = '{:.1f} mo'.format(days/30)
        if ret == '12.0 mo':
            ret = '1.0 y'
        return ret
    if days >= 7:
        ret = '{:.1f} w'.format(days/7)
        return ret
    ret = '{:.1f} d'.format(days)
    if ret == '7.0 d':
        ret = '1.0 w'
    return ret


def plot_average_rate(ax, df0, ndays, countries, threshold=10):
    '''
    Plot history of n-day average of daily increase percentage of selected countries.
    
    df: DataFrame
    ndays: int
    countries: a list of strings (country or state names), or a number for top n
    '''
    if type(countries) == int:
        df = df0.sort_values(by=df0.columns[-1], ascending=False)[:countries]
    else:
        df = df0.loc[countries]

    for idx, row in df.iterrows():
        selrow = row[row >= threshold]
        diff = selrow.diff(ndays)
        averate = (selrow / (selrow - diff))**(1/ndays) - 1
        dt = doubling_time(averate[-1])
        region = region_label(idx)
        if averate[-1] < 0.099:
            label = '{} ({:.1f}%,  {})'.format(region, 100*averate[-1], dt)
        else:
            label = '{} ({}%,  {})'.format(region, int(100*averate[-1]), dt)
        ax.plot_date(selrow.index, 100*averate, fmt='-', label=label, lw=3, alpha=0.7);
        
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)
        
    ax.set_ylim(-1,101)
    #plt.yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='Country (rate, doubling time)')
    plt.setp(legend.get_title(), fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Rate (%)', fontsize='xx-large');
    ax.set_title('Confirmed case daily increase rate ({}-day average) (update {})'.format(ndays, df.columns[-1]), fontsize='xx-large')


def plot_cumulated_since(ax, df0, countries, threshold=100, yscale='log'):
    '''
    Plot cumulated cases since the number of cases is confirmed
    
    df: DataFrame
    countries: a list of strings (country or state names), or a number for top n
    '''
    if type(countries) == int:
        df = df0.sort_values(by=df0.columns[-1], ascending=False)[:countries]
    else:
        df = df0.loc[countries]
        
    for j, (idx, row) in enumerate(df.iterrows()):
        selrow = row[row >= threshold]
        region = region_label(idx)
        label = '{} ({})'.format(region, int(selrow[-1]))
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
    ax.set_title('Confirmed cases vs. days since {}-th case (update {})'.format(threshold, df.columns[-1]), fontsize='xx-large')



