from collections import namedtuple, OrderedDict
from enum import Enum
import random
import pandas as pd
import numpy as np
from numpy import std, mean, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
from datetime import datetime, timedelta

import createDBTables

ActionEnum = Enum('Action', ['create_table', 'insert_to_db', 'run_paired_t_test'])

_alpha = 0.05

stats_to_compute = {'Goal difference':
                        namedtuple('Goal_Diff',
                                   ['order', 'integral_stat', 'title', 'home_label', 'away_label',
                                    'home_metric', 'away_metric'])
                         (0, True, 'Goal difference', 'Home Goal difference', 'Away Goal difference',
                         lambda ds: ds['home_team_goal_count'],
                         lambda ds: ds['away_team_goal_count']),

                    'Points difference':
                        namedtuple('Points_Diff',
                                   ['order', 'integral_stat', 'title', 'home_label', 'away_label',
                                    'home_metric', 'away_metric'])
                         (4, True, 'Points difference', 'Home Points difference', 'Away Points difference',
                         lambda ds: ds['Home points'],
                         lambda ds: ds['Away points']),

                    'Shot difference':
                        namedtuple('Shot_Diff',
                                   ['order', 'integral_stat', 'title', 'home_label', 'away_label',
                                    'home_metric', 'away_metric'])
                        (1, True, 'Shot difference', 'Home Shot difference', 'Away Shot difference',
                         lambda ds: ds['home_team_shots'],
                         lambda ds: ds['away_team_shots']),

                    'Shot accuracy':
                        namedtuple('Shot_Accuracy',
                                   ['order', 'integral_stat', 'title', 'home_label', 'away_label', 'home_metric',
                                    'away_metric'])
                        (2, False, 'Shot accuracy', 'Home Shot accuracy', 'Away Shot accuracy',
                         lambda ds: ds['Home Shot accuracy'],
                         lambda ds: ds['Away Shot accuracy']),

                    'Possession difference':
                        namedtuple('Possession',
                                   ['order', 'integral_stat', 'title', 'home_label', 'away_label', 'home_metric',
                                    'away_metric'])
                        (3, False, 'Possession difference', 'Home Possession difference', 'Away Possession difference',
                         lambda ds: ds['home_team_possession'],
                         lambda ds: ds['away_team_possession']),


                    }


"""
                            # Look at shot accuracy
                            'Shot accuracy':
                                namedtuple('Shot_Accuracy', ['home_label', 'away_label', 'home_metric', 'away_metric'])
                                ('Home Shot accuracy', 'Away Shot accuracy',
                                 lambda ds: ds['home_team_shots_on_target'] / ds['home_team_shots']
                                 if ds['home_team_shots'] else 0,
                                 lambda ds: ds['away_team_shots_on_target'] / ds['away_team_shots']
                                 if ds['away_team_shots'] else 0)
    
                            # Look at goals as a % of shots on target -- call it shot conversion rate
                            # or goals per shot
"""


def is_interactive():
    # print(globals())
    return __name__ == '__main__' and '__file__' and '__IPYTHON__' in globals()
    # return not hasattr(__main__, '__file__')


def generate_all_pairs(df, season):
    result = []
    teams = df[df['season'] == season].home_team_name.unique()
    teams = teams.tolist()
    print(teams)
    random.seed(0)
    r = random.random()
    random.shuffle(teams)
    print(teams)

    while len(teams) > 1:
        home = teams.pop()
        for away in teams:
            result.append((home, away))
    return result


def generate_all_pairs_evenly(df, season):
    """

    :param df:  the matches dataframe
    :param season: the season year to use
    :return: a list of tuples representing the pairs - home team is first, away team second
    """
    teams = df[df['season'] == season].home_team_name.unique()
    teams = teams.tolist()
    num_teams = len(teams)
    print(teams)
    column_names = teams
    row_names = teams

    # Create a matrix of all pair-wise combinations
    matrix = np.ones((len(teams), len(teams)), dtype=int)

    # Convert the matrix to data frame using the team names as both row and column labels
    pair_df = pd.DataFrame(matrix, columns=column_names, index=row_names)

    # Zero out self matches, i.e. the matches where a team plays itself which are obviously impossible
    for t in teams:
        pair_df.at[t, t] = 0

    # For 20 teams, this is 190
    max_pairs = ((num_teams ** 2) - num_teams) / 2
    num_pairs = 0
    res = []
    # Heuristically determined value. For 20 teams, ten will have 10 home games and ten will have 9 home games
    pairs_at_a_time = 10

    while num_pairs < max_pairs:
        for team in teams:
            for i in range(pairs_at_a_time):
                if pair_df.loc[team].max() == 0:
                    continue
                opp = pair_df.loc[team].idxmax()
                pair_df.at[team, opp] = 0
                pair_df.at[opp, team] = 0
                res.append((team, opp))
                num_pairs += 1

    print(f"Found: {len(res)} match pairs")
    return res


def count_home_games(pairs):
    home_games = {}
    for h, a in pairs:
        if h in home_games:
            home_games[h] += 1
        else:
            home_games[h] = 1

    for k in home_games:
        print(f"{k}: {home_games[k]}")


def paired_results_old(df, season, pairs):
    data = {'Home': [],
            'Away': [],
            'Home Goal difference': [],
            'Away Goal difference': []}

    col_order = ['Home', 'Away', 'Home Goal difference', 'Away Goal difference']

    for team, opponent in pairs:
        data['Home'].append(team)
        data['Away'].append(opponent)
        # Find home match
        home_match = df[(df['season'] == season) & (df['home_team_name'] == team) &
                        (df['away_team_name'] == opponent)]
        # print(home_match)
        home_match_series = home_match.iloc[0]
        home_match_goal_diff = home_match_series['home_team_goal_count'] - home_match_series['away_team_goal_count']
        # Find away match
        away_match = df[(df['season'] == season) & (df['home_team_name'] == opponent) &
                        (df['away_team_name'] == team)]
        away_match_series = away_match.iloc[0]
        away_match_goal_diff = away_match_series['away_team_goal_count'] - away_match_series['home_team_goal_count']

        data['Home Goal difference'].append(home_match_goal_diff)
        data['Away Goal difference'].append(away_match_goal_diff)

    df_pairs = pd.DataFrame(data)
    df_pairs = df_pairs[col_order]  # It's critical to specify the column order or the columns are displayed randomly

    return df_pairs


def print_match_pair(df, season, home_team, away_team):
    home_match = df[(df['season'] == season) & (df['home_team_name'] == home_team) &
                    (df['away_team_name'] == away_team)]
    # print(home_match)
    home_match_series = home_match.iloc[0]
    # Find away match
    away_match = df[(df['season'] == season) & (df['home_team_name'] == away_team) &
                    (df['away_team_name'] == home_team)]
    away_match_series = away_match.iloc[0]
    print(home_match_series)
    print(away_match_series)


def paired_results(df, season, pairs):
    home_match_series = None
    away_match_series = None

    stats_ordered = OrderedDict(sorted(stats_to_compute.items(), key=lambda t: t[1].order))
    print(stats_ordered)

    data = {'Home': [],
            'Away': [], }

    col_order = ['Home', 'Away', ]

    # Dynamically add the labels and data columns to the data frame based on the stats_ordered dict
    for k in stats_ordered:
        tupe = stats_ordered[k]
        for f in tupe._fields:
            if 'label' in f:
                data[getattr(tupe, f)] = []  # Gets the attribute named f from the tuple
                col_order.append(getattr(tupe, f))

    for team, opponent in pairs:
        data['Home'].append(team)
        data['Away'].append(opponent)
        # Find home match
        home_match = df[(df['season'] == season) & (df['home_team_name'] == team) &
                        (df['away_team_name'] == opponent)]
        # print(home_match)
        home_match_series = home_match.iloc[0]
        # Find away match
        away_match = df[(df['season'] == season) & (df['home_team_name'] == opponent) &
                        (df['away_team_name'] == team)]
        away_match_series = away_match.iloc[0]
        for k, v in stats_ordered.items():
            # print(f"key: {k} stat.first: {v.first} stat.second: {v.second}")
            home_diff = v.home_metric(home_match_series) - v.away_metric(home_match_series)
            away_diff = v.away_metric(away_match_series) - v.home_metric(away_match_series)
            data[v.home_label].append(home_diff)
            data[v.away_label].append(away_diff)

    df_pairs = pd.DataFrame(data)
    df_pairs = df_pairs[col_order]  # It's critical to specify the column order or the columns are displayed randomly

    return df_pairs


def run_paired_t_test_for_statistic(df_pairs, statistic):
    """

    :param df_pairs:
    :param statistic:
    :return: data frame representing the results
    """
    data = {
        'name': [],
        'value' : []
    }
    col_order = ['name', 'value']
    home = "Home " + statistic
    away = "Away " + statistic

    print(f"T-test for {statistic}")
    (statistic, pvalue) = t_test = stats.ttest_rel(df_pairs[home], df_pairs[away])
    statistic = format(statistic, ".6g")
    pvalue = format(pvalue, ".6g")
    print(t_test)

    data['name'].append('Null hypothesis:')
    data['value'].append(r'$\mu$$_{home}$ = $\mu$$_{away}$')
    data['name'].append('Alt. hypothesis:')
    data['value'].append(r'$\mu$$_{home}$ != $\mu$$_{away}$')
    data['name'].append("t statistic:")
    data['value'].append(f"{statistic}")
    data['name'].append("p-value:")
    data['value'].append(f"{pvalue}")
    data['name'].append("   ")
    data['value'].append("   ")
    if float(pvalue) < _alpha:
        data['name'].append(fr'p-value < $\alpha$ ({_alpha})')
        data['value'].append("We reject the\nnull hypothesis")
    else:
        data['name'].append(fr'p-value >= $\alpha$ ({_alpha})')
        data['value'].append("We fail to reject the\nnull hypothesis")
    df = pd.DataFrame(data)
    df = df[col_order]
    # df.set_index('name', inplace=True)
    print(tabulate(df, tablefmt='psql'))
    return df


def home_shot_accuracy(series):
    if series['home_team_shots']:
        accuracy = series['home_team_shots_on_target'] / series['home_team_shots']
    else:
        accuracy = 0
    return accuracy


def away_shot_accuracy(series):
    if series['away_team_shots']:
        accuracy = series['away_team_shots_on_target'] / series['away_team_shots']
    else:
        accuracy = 0
    return accuracy


def home_goals_per_shot(series):
    if series['home_team_shots']:
        goals_per_shot = series['home_team_goal_count'] / series['home_team_shots']
    else:
        goals_per_shot = 0
    return goals_per_shot


def away_goals_per_shot(series):
    if series['away_team_shots']:
        goals_per_shot = series['away_team_goal_count'] / series['away_team_shots']
    else:
        goals_per_shot = 0
    return goals_per_shot


def home_points(series):
    if series['home_team_goal_count'] == series['away_team_goal_count']:
        points = 1
    elif series['home_team_goal_count'] > series['away_team_goal_count']:
        points = 3
    else:
        points = 0
    return points


def away_points(series):
    if series['home_team_goal_count'] == series['away_team_goal_count']:
        points = 1
    elif series['home_team_goal_count'] > series['away_team_goal_count']:
        points = 0
    else:
        points = 3
    return points

def augment_df(match_df):
    match_df['Home Shot accuracy'] = match_df.apply(home_shot_accuracy, axis=1)
    match_df['Away Shot accuracy'] = match_df.apply(away_shot_accuracy, axis=1)

    match_df['Home Shots per goal'] = match_df.apply(home_goals_per_shot, axis=1)
    match_df['Away Shots per goal'] = match_df.apply(away_goals_per_shot, axis=1)

    match_df['Home points'] = match_df.apply(home_points, axis=1)
    match_df['Away points'] = match_df.apply(away_points, axis=1)

    # Add output of describe to a new data frame
    descriptive_stats = pd.DataFrame({'Home Shot accuracy': match_df['Home Shot accuracy'].describe()})
    descriptive_stats['Away Shot accuracy'] = match_df['Away Shot accuracy'].describe()
    # descriptive_stats['Home Shots per goal'] = match_df[match_df['season'] == season]['Home Shots per goal'].describe()
    # descriptive_stats['Away Shots per goal'] = match_df[match_df['season'] == season]['Away Shots per goal'].describe()
    print(tabulate(descriptive_stats, headers='keys', tablefmt='psql'))



def descriptive_stats_to_df(match_df, season, statistic):
    """
    Returns the home and away descriptive statistics for a given season as a data frame

    :param match_df:
    :param season:
    :param statistic:
    :return:
    """
    descriptive_stats = pd.DataFrame()

    stat_metrics = [statistic.home_metric, statistic.away_metric]

    for metric in stat_metrics:
        ds = metric(match_df[match_df['season'] == season]).describe()

        if not descriptive_stats.empty:
            descriptive_stats[ds.name] = metric(match_df[match_df['season'] == season]).describe()
        else:
            descriptive_stats = pd.DataFrame({ds.name: metric(match_df[match_df['season'] == season]).describe()})
    return descriptive_stats


def plot_statistic(plot_ax, df_pairs, year, statistic):
    """
    If the statistic has integral or unit level x-axis values then the number of bins is the number of x-axis units
    Else plots the histogram using about 20 bins
    :param plot_ax:
    :param df_pairs:
    :param year:
    :param statistic:
    :return:
    """
    num_bins = 20
    home = statistic.home_label
    away = statistic.away_label
    max_x = max(df_pairs[home].max(), df_pairs[away].max()) + 0.5
    min_x = min(df_pairs[home].min(), df_pairs[away].min()) - 0.5

    # Assume about 20 bins
    # 20 = (max-min) * step. Therefore step = 20 / (max - min)
    if statistic.integral_stat:
        step = 1
    else:
        step = (max_x - min_x) / num_bins

    # +1 because the high of range is not included
    bins = np.arange(min_x, max_x + 1, step).tolist()

    # while len(bins) > 20:
    #    bins = np.arange(min_x, max_x + 1, i).tolist()
    #    i += 1

    x = df_pairs[home]
    y = df_pairs[away]

    plot_ax.set_title(statistic.title + f" ({year})")
    # plot_ax.hist([x, y], bins, label=[home, away])

    sea_ax = sns.distplot(x, bins, hist=True, kde=True, norm_hist=False, axlabel=False, color='r', ax=plot_ax)
    sns.distplot(y, bins, hist=True, kde=True, norm_hist=False, axlabel=False, color='b', ax=plot_ax)
    sea_ax.set(xlabel=statistic.title, ylabel='Probability Density')
    plot_ax.legend([home.split(' ', 1)[0], away.split(' ', 1)[0]], loc='upper right')


def calc_cohens_d(group1, group2):
    # Compute Cohen's d.
    # returns a floating point number

    diff = group1['mean'] - group2['mean']
    n1 = group1['count']
    n2 = group2['count']
    var1 = group1['std']**2
    var2 = group2['std']**2

    # Calculate the pooled threshold as shown earlier
    pooled_var = ((n1-1) * var1 + (n2-1) * var2) / (n1 + n2 - 1 - 1)

    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    return d

def calc_cohens_d_for_paired_samples(group1, group2):
    # Compute Cohen's d.
    # returns a floating point number

    diff = group1['mean'] - group2['mean']
    n1 = group1['count']
    n2 = group2['count']
    var1 = group1['std']**2
    var2 = group2['std']**2

    # Calculate the pooled threshold as shown earlier
    pooled_var = (var1 + var2) / 2

    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    return d


def plot_descriptive_statistic(match_df, season, statistic, plot_ax):
    """

    :param match_df:
    :param season:
    :param statistic
    :param plot_ax:
    :return:
    """
    df = descriptive_stats_to_df(match_df, season, statistic)
    stat_metrics = [statistic.home_metric, statistic.away_metric]

    cohens_d = calc_cohens_d(df.iloc[:, 0], df.iloc[:, 1])
    print(f"Cohen's d: {cohens_d}")
    cohens_d = calc_cohens_d_for_paired_samples(df.iloc[:, 0], df.iloc[:, 1])
    print(f"Cohen's d for paired samples: {cohens_d}")

    df = df.round(3)
    # Only care about mean and std deviation
    df = df.loc[['mean', 'std']]
    # Replace underscores with spaces in column names
    new_cols = []
    for c in df.columns:
        new_col = c.replace('_', ' ')
        new_col = new_col.title()
        new_cols.append(new_col)
    df.columns = new_cols

    print(df)
    mpl_table = plot_ax.table(cellText=df.values, rowLabels=df.index,  colLabels=df.columns, loc='center',
                              edges='horizontal')
    mpl_table.auto_set_font_size(False)
    mpl_table.scale(1, 1.1)
    return cohens_d


def plot_t_test_result(df, cohens_d, plot_ax):
    plot_ax.axis('off')

    title = fr'Two sample paired T-test: $\alpha$={_alpha}'

    data = {
        'name': [],
        'value': []
    }
    # Commented out Cohen's D
    # data['name'].append("Cohen's D:")
    # data['value'].append(format(cohens_d, ".3g"))
    df = pd.concat([pd.DataFrame(data),df])

    plot_ax.set_title(title)
    # Left, bottom, width, height  bbox=[0.0, 0, 3, 2]
    mpl_table = plot_ax.table(cellText=df.values,  loc='center', edges='open', cellLoc='left')

    # colWidths=[0.4]*len(df.columns))
    mpl_table.auto_set_font_size(False)
    mpl_table.scale(1, 1.5)


def plot_figure(match_df, df_pairs, year, t_test_results, stats: list, title):
    """
    Plot up to two stats on a figure
    :param match_df:
    :param df_pairs:
    :param year:
    :param t_test_results:
    :param stats:
    :return:
    """
    num_stats = 2
    i = 0
    fig, ax_lst = plt.subplots(3, 2, figsize=(11, 7.5))
    fig.suptitle(title, fontsize=14)

    # Turn of the axis display for the bottom 4 charts in the figure, using list comprehension
    flat_list = [j.axis('off') for sub in ax_lst[1:] for j in sub]

    while i < num_stats and i < len(stats):
        stat = stats[i]
        plot_statistic(ax_lst[0, i], df_pairs, year, stat)
        cohens_d = plot_descriptive_statistic(match_df, year, stat, ax_lst[1, i])
        plot_t_test_result(t_test_results[stat.title], cohens_d, ax_lst[2, i])
        i += 1
    plt.show()


def run_paired_t_test_with_pairs(match_df, year, pairs, title):
    t_test_results = {}

    """  Moving this
    match_df = createDBTables.read_matches_from_db()
    # Add derived metrics to the data frame
    augment_df(match_df)
    
    """


    print(len(pairs))
    df_pairs = paired_results(match_df, year, pairs)

    tests_to_run = ['Goal difference', 'Shot difference', 'Possession difference', 'Points difference']
    for test in tests_to_run:
        t_test_result = run_paired_t_test_for_statistic(df_pairs, test)
        t_test_results[test] = t_test_result

    plot_figure(match_df, df_pairs, year, t_test_results,
                [stats_to_compute['Goal difference'], stats_to_compute['Possession difference']],
                title)

    # plot_figure(match_df, df_pairs, year, t_test_results,
    #            [stats_to_compute['Shot difference'], stats_to_compute['Points difference']])

    if False:
        print(df_pairs['Home Shot difference'].describe())
        print(df_pairs['Away Shot difference'].describe())
        print("T-test for shots difference")
        print(stats.ttest_rel(df_pairs['Home Shot difference'], df_pairs['Away Shot difference']))
        print(tabulate(df_pairs, headers='keys', tablefmt='psql'))
        print_match_pair(match_df, year, 'Manchester United', 'Liverpool')

    return match_df, df_pairs


def run_paired_t_test(year):
    match_df = createDBTables.read_matches_from_db()
    # Add derived metrics to the data frame
    augment_df(match_df)
    pairs = generate_all_pairs_evenly(match_df, year)
    return run_paired_t_test_with_pairs(match_df, year, pairs, f"Paired T-test for all {len(pairs)} paired games")


def get_corresponding_away_game_for_single_game(match_df, home_game_key):
    """
    Should return a list of paired games
    :return:
    """
    res = []
    home_match = match_df[match_df['match_key'] == home_game_key]
    home_match_series = home_match.iloc[0]

    home_team = home_match_series['home_team_name']
    away_team = home_match_series['away_team_name']
    season = home_match_series['season']

    res.append((home_team, away_team))
    return res


def get_bad_weather_pairs(match_df):
    """
    Currently just hard-coded for testing purposes
    :param match_df
    :param year:
    :return:
    """
    pairs = []

    # The match keys for the bad weather home games
    bad_weather_home_games = ['Wolverhampton Wanderers vs Liverpool 12-21-18',
                              'AFC Bournemouth vs Brighton & Hove Albion 12-22-18']
    for game in match_df['match_key']:
        print(game)
        pairs.extend(get_corresponding_away_game_for_single_game(match_df, game))
    print(pairs)
    return pairs


bad_weather = ['Blizzard', 'Heavy rain', 'Light sleet', 'Light snow',
               'Light snow showers', 'Mist', 'Moderate or heavy rain with thunder',
               'Moderate rain', 'Moderate rain at times']


def get_hour(series):
    # Get the time of half time by adding 45 minutes
    dt = series['datetime'] + timedelta(minutes=45)
    # Round to nearest hour, times on the half hour round up
    if dt.minute >= 30:
        dt += timedelta(minutes=30)
    return dt.hour * 100


def get_weather_for_match_hour(match_df, weather_df, season):
    match_df['hour'] = match_df.apply(get_hour, axis=1)
    cols = list(match_df)
    cols.insert(2, cols.pop(cols.index('hour')))
    match_df = match_df.loc[:, cols]

    merged_df = pd.merge(match_df[match_df['season'] == season], weather_df, how='left',
                         left_on=['match_key', 'hour'],
                         right_on=['match_key', 'time'])
    return merged_df


def bad_weather_matches(merged_df):
    bad_weather_desc = merged_df[merged_df['weatherDesc'].isin(bad_weather)]
    bad_weather_matches = merged_df[(merged_df['precipMM'] >= 0.4) |
                                    (merged_df['visibilityMiles'] < 1) |
                                    (merged_df['tempF'] <= 32) |
                                    (merged_df['windspeedMiles'] >= 25)]
    bw_concat = pd.concat([bad_weather_desc, bad_weather_matches]).drop_duplicates().reset_index(drop=True)

    print(
        f"Lengths: bad_weather_desc: {len(bad_weather_desc)}, bad_weather_matches: {len(bad_weather_matches)}, bw_concat: {len(bw_concat)}")
    return bw_concat


def get_selected_pairs_in_df(pairs, season, df):
    res = pd.DataFrame()
    for team, opponent in pairs:

        # Find home match
        home_match = df[(df['season'] == season) & (df['home_team_name'] == team) &
                        (df['away_team_name'] == opponent)]

        # Find away match
        away_match = df[(df['season'] == season) & (df['home_team_name'] == opponent) &
                        (df['away_team_name'] == team)]
        res = pd.concat([res, home_match, away_match])
    return res


def run_bad_weather_paired_t_test(year):
    """
    Assumption should be that there is just bad weather for the home match
    :param year:
    :return:
    """
    # print(tabulate(bad_weather_df, headers='keys', tablefmt='psql'))
    match_df = createDBTables.read_matches_from_db()
    augment_df(match_df)
    weather_df = createDBTables.read_weather_from_db()
    merged_df = get_weather_for_match_hour(match_df, weather_df, 2018)
    bad_weather_df = bad_weather_matches(merged_df)

    print(f"Number of bad weather matches: {len(bad_weather_df)}")

    pairs = get_bad_weather_pairs(bad_weather_df)

    # get subset dataframe just with the pairs of bad weather/good weather matches
    subset_match_df = get_selected_pairs_in_df(pairs, 2018, match_df)
    print(tabulate(subset_match_df, headers='keys', tablefmt='psql'))
    return run_paired_t_test_with_pairs(subset_match_df, year, pairs,
                                        f"Paired t-test for {int(len(subset_match_df)/2)} bad weather home games")


def run_action(year):
    action = ActionEnum.run_paired_t_test

    if action == ActionEnum.create_table:
        createDBTables.open_conn_and_create_main_table()
    elif action == ActionEnum.insert_to_db:
        createDBTables.open_db()
        createDBTables.read_all_csv_and_insert()
        createDBTables.close_db()
    elif action == ActionEnum.run_paired_t_test:
        return run_paired_t_test(year)


if not is_interactive():
    run_paired_t_test(2018)

    # run_bad_weather_paired_t_test(2018)
