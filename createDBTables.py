import re
from enum import Enum

import pandas as pd
from datetime import datetime
import config


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


if not is_interactive():
    import pymysql as db_con
else:
    import mysql.connector as db_con


cnx = None
cursor = None




def open_db():
    global cnx
    global cursor

    cnx = db_con.connect(
        host=config.host,
        user=config.user,
        passwd=config.pw,
        database='soccer_matches'
    )

    cursor = cnx.cursor()


def close_db():
    cursor.close()
    cnx.close()


def create_match_table():
    sql = """
    CREATE TABLE matches ( 
    match_key  varchar(150),
    datetime  datetime,
    season  int,
    timestamp  int,
    date_GMT    varchar(50),
    status      varchar(50),
    attendance  int,
    home_team_name varchar(50),
    away_team_name varchar(50),
    home_team_goal_count int,
    away_team_goal_count int,
    total_goal_count int,
    total_goals_at_half_time int,
    home_team_goal_count_half_time int,
    away_team_goal_count_half_time int,
    home_team_goal_timings  varchar(100),
    away_team_goal_timings  varchar(100),
    home_team_corner_count  int,
    away_team_corner_count  int,
    home_team_yellow_cards  int,
    home_team_red_cards     int,
    away_team_yellow_cards  int,
    away_team_red_cards     int,
    home_team_shots         int,
    away_team_shots         int,
    home_team_shots_on_target  int,
    away_team_shots_on_target  int,
    home_team_shots_off_target  int,
    away_team_shots_off_target  int,
    home_team_fouls   int,
    away_team_fouls   int,
    home_team_possession   int,
    away_team_possession   int,
    stadium_name    varchar(100),
    primary key(match_key)
    )"""

    cursor.execute(sql)


def add_to_table(movie_dict):
    for movie_id in movie_dict.keys():
        data_movie = {
            'movie_id': movie_id,
            'title': movie_dict[movie_id][0],
            'user_rating': movie_dict[movie_id][1],
            'metascore': movie_dict[movie_id][2],
        }
        # print(data_movie)

        inserting = """INSERT INTO Master_Table  
         (movie_id, title, user_rating, metascore) 
         VALUES (%(movie_id)s, %(title)s, %(user_rating)s, %(metascore)s);"""

        cursor.execute(inserting, data_movie)
        cnx.commit()


def insert_to_table(df):
    """
    This method assumes there is no index column on the data frame
    :param df:
    :return:
    """
    global cnx, cursor
    debug = False
    # creating column list for insertion
    cols = ", ".join([str(i) for i in df.columns.tolist()])

    # Insert DataFrame records one by one.
    for i, row in df.iterrows():
        # Convert datetime obj to string
        row['datetime'] = row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        # Replace N/A attendance with -1
        row['attendance'] = (-1 if row['attendance'] == 'N/A' else row['attendance'])
        sql = "INSERT INTO matches (" + cols + ") VALUES (" + "%s," * (len(row) - 1) + "%s)"
        if debug:
            print(sql)
            print(tuple(row))
        cursor.execute(sql, tuple(row))
        # the connection is not auto-committed by default, so we must commit to save our changes
        cnx.commit()


def open_conn_and_create_main_table():
    open_db()
    create_match_table()
    close_db()


def parse_date(series):
    """
    E.g 'Jun 28 2018  18:40:00'
    """
    dt_str = series['date_GMT']
    dt_str = dt_str.replace('-', '')
    date_time_obj = datetime.strptime(dt_str, '%b %d %Y %I:%M%p')
    # time_str = datetime.strftime(date_time_obj, "%H:%M:%S")
    return date_time_obj


def create_key(series):
    dtime = series['datetime'].date()
    date = datetime.strftime(dtime, "%m-%d-%y")
    home_team = series['home_team_name']
    away_team = series['away_team_name']
    key = home_team + " vs " + away_team + " " + date
    return key


def fix_stadiums(series):
    fix_stadium = {"Vicarage Road Stadium": "Vicarage Road",
                   "Bet365 Stadium": "Britannia Stadium",
                   "bet365 Stadium": "Britannia Stadium",
                   "The DW Stadium": "DW Stadium",
                   "St Andrew's Trillion Trophy Stadium": "St. Andrew's Stadium",
                   "The American Express Community Stadium": "American Express Community Stadium",
                   "John Smith's Stadium": "The John Smith's Stadium",
                   "KCOM Stadium": "KC Stadium",
                   "Loftus Road Stadium": "Loftus Road",
                   "St. Mary's Stadium": "St Mary's Stadium",
                   }

    stad_keys = fix_stadium.keys()

    res = re.sub(r"\(.*\)", "", series['stadium_name'])
    res = res.strip()

    if res in stad_keys:
        return fix_stadium[res]
    else:
        return res


def read_csv(filename):
    # 3rd arg means no NaNs will be generated
    df = pd.read_csv(filename, sep=',', keep_default_na=False)
    return df


def save_table_to_csv():
    df = pd.read_sql("select * from matches", cnx)
    df.to_csv('all_matches.csv', sep=',')


def read_csv_and_insert_db(season, filename):
    match_dir = 'match_data/'
    df = read_csv(match_dir + filename)

    # Drop the betting columns
    cols = df.columns
    cols = [c for c in cols if not 'pre-match' in c.lower()]
    cols = [c for c in cols if not 'pre_match' in c.lower()]
    cols = [c for c in cols if not 'ppg' in c.lower()]
    cols = [c for c in cols if not c.lower().startswith('odds')]
    cols = [c for c in cols if not c.lower().startswith('over')]
    # print(cols)
    df = df[cols]

    df['datetime'] = df.apply(parse_date, axis=1)
    df['match_key'] = df.apply(create_key, axis=1)
    df['season'] = season
    df['stadium_name'] = df.apply(fix_stadiums, axis=1)
    cols = list(df)
    # move the column to head of list using index, pop and insert
    cols.insert(0, cols.pop(cols.index('season')))
    cols.insert(0, cols.pop(cols.index('datetime')))
    cols.insert(0, cols.pop(cols.index('match_key')))

    df = df.loc[:, cols]
    df.set_index('match_key')
    print(f"Filename: {filename} rowcount: {len(df.index)}")
    insert_to_table(df)


def read_all_csv_and_insert():
    files = {
        2009: 'england-premier-league-matches-2009-to-2010-stats.csv',
        2010: 'england-premier-league-matches-2010-to-2011-stats.csv',
        2011: 'england-premier-league-matches-2011-to-2012-stats.csv',
        2012: 'england-premier-league-matches-2012-to-2013-stats.csv',
        2013: 'england-premier-league-matches-2013-to-2014-stats.csv',
        2014: 'england-premier-league-matches-2014-to-2015-stats.csv',
        2015: 'england-premier-league-matches-2015-to-2016-stats.csv',
        2016: 'england-premier-league-matches-2016-to-2017-stats.csv',
        2017: 'england-premier-league-matches-2017-to-2018-stats.csv',
        2018: 'england-premier-league-matches-2018-to-2019-stats.csv'}

    for year in files:
        read_csv_and_insert_db(year, files[year])


def read_matches_from_db():
    open_db()
    df = pd.read_sql("select * from matches", cnx)
    close_db()
    return df


def read_weather_from_db():
    open_db()
    weather_df = pd.read_sql("select * from weather", cnx)
    close_db()
    return weather_df
