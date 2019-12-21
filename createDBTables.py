import pandas as pd
from datetime import datetime
import config

if config.pycharm:
    import pymysql
else:
    import mysql.connector

cnx = None
cursor = None


def open_db():
    global cnx
    global cursor

    if config.pycharm:
        cnx = pymysql.connect(
            host=config.host,
            user=config.user,
            passwd=config.pw,
            database='soccer_matches'
        )
    else:
        cnx = mysql.connector.connect(
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


def read_csv(filename):
    # 3rd arg means no NaNs will be generated
    df = pd.read_csv(filename, sep=',', keep_default_na=False)
    return df


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


create_table = False

if create_table:
    open_conn_and_create_main_table()
else:
    open_db()
    read_all_csv_and_insert()
    close_db()
