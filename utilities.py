import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
import re, warnings

us_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
            'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 
            'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 
            'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
            'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 
            'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico',
            'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 
            'Utah', 'Vermont', 'Virgin Islands', 'Virginia', 'Washington', 
            'West Virginia', 'Wisconsin', 'Wyoming', 'American Samoa',
            'Northern Mariana Islands']
state_abbr = dict(AL='Alabama', AK='Alaska', AZ='Arizona', AR='Arkansas', 
                  CA='California', CO='Colorado', CT='Connecticut', DE='Delaware', 
                  DC='District of Columbia', FL='Florida', GA='Georgia', HI='Hawaii', 
                  ID='Idaho', IL='Illinois', IN='Indiana', IA='Iowa', KS='Kansas',
                  KY='Kentucky', LA='Louisiana', ME='Maine', MD='Maryland', 
                  MA='Massachusetts', MI='Michigan', MN='Minnesota', MS='Mississippi',
                  MO='Missouri', MT='Montana', NE='Nebraska', NV='Nevada',
                  NH='New Hampshire', NJ='New Jersey', NM='New Mexico', NY='New York',
                  NC='North Carolina', ND='North Dakota', OH='Ohio', OK='Oklahoma',
                  OR='Oregon', PA='Pennsylvania', RI='Rhode Island', SC='South Carolina',
                  SD='South Dakota', TN='Tennessee', TX='Texas', UT='Utah', VT='Vermont',
                  VA='Virginia', WA='Washington', WV='West Virginia', WI='Wisconsin',
                  WY='Wyoming')
state_abbr['D.C.'] = state_abbr['DC']


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

    
def plot_cumulated_histories(ax, df, i0=0, i1=10, title=None, yscale='log', ymax=None, case='confirmed cases', plot_remaining=False, starting_date=None, legend_title='Country'):
    '''
    Plot the history of countries that have the i0-th to i10-th highest confirmed case
    
    ax: axes
    df: dataframe
    i0, i1: (i0+1)-th to i1-th highest are ploted.
    case: 'confirmed cases' or 'deaths'
    '''
    dfp = df.sort_values(by=df.columns[-1], ascending=False)
    for j, (idx, row) in enumerate(dfp[i0:i1].iterrows()):
        facecolor=None
        if j < 10:
            lst = '-'
        elif j < 20:
            lst = '--'
            facecolor='none'
        else:
            lst = ':'
            facecolor='none'
        region = region_label(idx)
        label = '{} ({})'.format(region, int(row[-1]))
        ax.plot_date(row.index, row, fmt=lst, label=label, lw=3, alpha=0.8);
        
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
        legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (# deaths)'.format(legend_title))
    else:
        legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (# cases)'.format(legend_title))
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
    for j, (idx, row) in enumerate(dfp[i0:i1].iterrows()):
        facecolor=None
        if j < 10:
            lst = '-'
        elif j < 20:
            lst = '--'
            facecolor='none'
        else:
            lst = ':'
            facecolor='none'
        region = region_label(idx)
        label = '{} ({:.2f}%)'.format(region, row[-1])
        sel = dfc.loc[idx] >= threshold
        ax.plot_date(row.loc[sel].index, row.loc[sel], fmt=lst, label=label, lw=3, alpha=0.8);

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


def plot_average_rate(ax, df0, ndays, countries, threshold=10, legend_title='Country', title_head='Confirmed case'):
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

    for j, (idx, row) in enumerate(df.iterrows()):
        region = region_label(idx)
        selrow = row[row >= threshold]
        diff = selrow.diff(ndays)
        averate = (selrow / (selrow - diff))**(1/ndays) - 1
        dt = doubling_time(averate[-1])
        if averate[-1] < 0.099:
            label = '{} ({:.1f}%,  {})'.format(region, 100*averate[-1], dt)
        else:
            label = '{} ({}%,  {})'.format(region, int(100*averate[-1]), dt)

        facecolor=None
        if j < 10:
            lst = '-'
        elif j < 20:
            lst = '--'
            facecolor='none'
        else:
            lst = ':'
            facecolor='none'

        ax.plot_date(selrow.index, 100*averate, fmt=lst, label=label, lw=3, alpha=0.7);
        
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    plt.setp(ax.get_xticklabels(), fontsize='x-large', rotation=35)
        
    ax.set_ylim(-1,101)
    #plt.yscale(yscale);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (rate, doubling time)'.format(legend_title))
    plt.setp(legend.get_title(), fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Rate (%)', fontsize='xx-large');
    ax.set_title('{} daily increase rate ({}-day average) (update {})'.format(title_head, ndays, df.columns[-1]), fontsize='xx-large')


def plot_cumulated_since(ax, df0, countries, threshold=100, yscale='log', legend_title='Country'):
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
        facecolor=None
        if j < 10:
            lst = '-'
        elif j < 20:
            lst = '--'
            facecolor='none'
        else:
            lst = ':'
            facecolor='none'
        ax.plot(selrow.values, 'C{}{}'.format(j%10,lst), label=label, lw=3, alpha=0.7);
        ax.scatter(len(selrow)-1, selrow.values[-1], s=50, marker='o', color='C{}'.format(j%10), linewidths=2, facecolor=facecolor)
        
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    ax.set_yscale(yscale)  
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (# cases)'.format(legend_title))
    plt.setp(legend.get_title(),fontsize='x-large')
    ax.grid(axis='y');
    ax.set_ylabel('Reported confirmed cases', fontsize='xx-large');
    ax.set_xlabel('Days since {}-th case'.format(threshold), fontsize='xx-large')
    ax.set_title('Confirmed cases vs. days since {}-th case (update {})'.format(threshold, df.columns[-1]), fontsize='xx-large')


def plot_new_vs_existing(ax, df0, ndays, countries, threshold=1, legend_title='Country', case_type='Confirmed case'):
    '''
    Plot confirmed cases in the past ndays vs existing cases

    df: DataFrame
    ndays: int
    countries: a list of strings (country or state names), or a number for top n
    threshold: 
    '''
    if type(countries) == int:
        df = df0.sort_values(by=df0.columns[-1], ascending=False)[:countries]
    else:
        df = df0.loc[countries]

    for j, (idx, row) in enumerate(df.iterrows()):
        selrow = row[row >= threshold]
        diff = selrow.diff(ndays)
        region = region_label(idx)
        label = '{} ({:d})'.format(region, int(selrow[-1]))
        facecolor=None
        if j < 10:
            lst = '-'
        elif j < 20:
            lst = '--'
            facecolor='none'
        else:
            lst = ':'
            facecolor='none'
        ax.plot(selrow, diff, 'C{}{}'.format(j%10,lst), label=label, lw=3, alpha=0.7)
        ax.scatter(selrow[-1], diff[-1], s=50, marker='o', color='C{}'.format(j%10), linewidths=2, facecolor=facecolor)


    plt.setp(ax.get_xticklabels(), fontsize='x-large')
    plt.setp(ax.get_yticklabels(), fontsize='x-large')
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    legend = ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left', fontsize='x-large', title='{} (#cases)'.format(legend_title))
    plt.setp(legend.get_title(), fontsize='x-large')
    ax.grid();
    ax.set_xlabel('Existing cases', fontsize='xx-large')
    ax.set_ylabel('New cases in past {} days'.format(ndays), fontsize='xx-large');
    ax.set_title('New {} in past {} days vs. existing cases (update {})'.format(case_type, ndays, df.columns[-1]), fontsize='xx-large')


def extract_state_name(string):
    '''
    Extract the full state name from a string
    '''
    if string.startswith('Unassigned') or string.startswith('Diamond Princess') or string.startswith('Grand Princess'):
        return None

    if string in ['Grand Princess', 'US', 'Wuhan Evacuee', 'Recovered']:
        return None

    if string == 'Chicago':
        return 'Illinois'

    if re.search('Virgin Islands', string):
        return 'Virgin Islands'

    fields = [x.strip() for x in string.split(',')]

    if len(fields) == 1:
        if fields[0] in us_states:
            return fields[0]
        else:
            print(fields)
            raise ValueError('Cannot get the state name from the string {}'.format(string))
    #
    fields2 = fields[1].split()
    sbr = fields2[0]
    try:
        ret = state_abbr[sbr]
    except:
        raise ValueError('Unknown abbreviation {} from the string {}'.format(sbr, string))
    return ret

def find_us_state_cases(dfdaily):
    '''
    Return 3 lists of confirmed, deaths, and recovered cases. 
    The order is based on us_states list

    dfdaily: daily state report DataFrame
    '''
    columns = dfdaily.columns
    if 'Country/Region' in columns:
        dfus = dfdaily.loc[dfdaily['Country/Region']=='US']
    elif 'Country_Region' in columns:
        dfus = dfdaily.loc[dfdaily['Country_Region']=='US']
    else:
        print(columns)
        raise ValueError('Which column is US in?')
    if 'Province/State' in columns:
        col_state = 'Province/State'
    elif 'Province_State' in columns:
        col_state = 'Province_State'
    else:
        print(columns)
        raise ValueError('Which column is states in?')

    #print(dfus.columns)
    confirmed = {}
    deaths = {}
    recovered = {}
    for sn in us_states:
        confirmed[sn] = 0
        deaths[sn] = 0
        recovered[sn] = 0
    for idx, row in dfus.iterrows():
        state = row[col_state]
        state_name = extract_state_name(state)
        if state_name is None:
            continue
        confirmed[state_name] += row['Confirmed']
        deaths[state_name] += row['Deaths']
        recovered[state_name] += row['Recovered']

    clist = [confirmed[k] for k in us_states]
    dlist = [deaths[k] for k in us_states]
    rlist = [recovered[k] for k in us_states]
    return clist, dlist, rlist


def us_states_dataframe_from_daily_report(start_date, end_date):
    '''
    Return 3 dataframes of states data (row index is state, column is date):
    confirmed df, deaths df, and recovered df.
    '''
    dtrange = pd.date_range(start_date, end_date)
    path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports'
    d_confirmed = {}
    d_deaths = {}
    d_recovered = {}
    for dt in dtrange:
        dstr = dt.strftime('%m-%d-%Y')
        url = path+'/{}.csv'.format(dstr)
        try:
            data = pd.read_csv(url)
        except:
            warnings.warn('Cannot read from URL {}'.format(url))
            continue
        confirmed, deaths, recovered = find_us_state_cases(data)
        d_confirmed[dstr] = confirmed
        d_deaths[dstr] = deaths
        d_recovered[dstr] = recovered

    df_confirmed = pd.DataFrame(data=d_confirmed, index=us_states)
    df_deaths = pd.DataFrame(data=d_deaths, index=us_states)
    df_recovered = pd.DataFrame(data=d_recovered, index=us_states)

    return df_confirmed, df_deaths, df_recovered


def update_us_states_dataframe(df_orig, end_date=None, case_type='confirmed'):
    '''
    Return a new dataframe of us states stats with update up to end_date:

    df_orig: the original dataframe
    end_date: end date. If None, set to today's date
    case_type: 'confirmed', 'deaths', or 'recovered'
    '''
    if end_date is None:
        end_date = date.today()

    # get the last date plus one day as the start date
    start_date = pd.to_datetime(df_orig.columns[-1]) + timedelta(days=1)
    #print(start_date, end_date)
    # additional data
    dconf, ddeaths, drecov = us_states_dataframe_from_daily_report(start_date, end_date) 
    if case_type == 'confirmed':
        df = dconf
    elif case_type == 'deaths':
        df = ddeaths
    elif case_type == 'recovered':
        df = drecov
    else:
        raise ValueError('unknown case_type {}'.format(case_type))

    result = pd.concat([df_orig, df], axis=1, sort=False)


    return result

