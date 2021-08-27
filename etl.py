import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek, date_format, monotonically_increasing_id



config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark



def process_song_data(spark, input_data, output_data):
    """
    Load data from song_data dataset and extract columns:
    - songs tables
    - artist tables  
    --> write both extracted data into parquet files which will be loaded on a data sotrage "s3".
    Parameters
    ----------
    spark: session
          This is the spark session that has been created
    input_data: path
           This is the path to the song_data into the data storage "s3 bucket".
    output_data: path
            This is the path to where the both parquet files (songs tables, artist tables) will be written.
    """
    
    # get filepath to song data file
    song_data = "./data/song_data/A/A/A/*.json"
    
    # read song data file
    df = spark.read.json(song_data)
    print('schema of song_data files')
    df.printSchema()
    # extract columns to create songs table 
    #songs table: song_id, title, artist_id, year, duration
    songs_table = df.selectExpr('song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates()
    print('First 10 rows of song_table')
    songs_table.show(10)
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id') \
                     .parquet(os.path.join(output_data, 'songs/songs.parquet'), 'overwrite')



    # extract columns to create artists table
    #artist table: artist_id, name, location, lattitude, longitude
    artists_table = df.selectExpr('artist_id', 'artist_name as name', 'artist_location as location',
                                  'artist_latitude as latitude', 'artist_longitude as logitude').dropDuplicates()
    print('First 10 rows of artists_table')
    artists_table.show(10)
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists/artists.parquet'), 'overwrite')
    




def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = "./data/log_data/*.json"

    # read log data file
    df = spark.read.json(log_data)
    print('schema of the log_data')
    df.printSchema()
    
    
    print('fitered actions')
    # filter by actions for song plays
    actions_df = df.filter(df.page == 'NextSong').selectExpr('ts', 'userId', 'level', 'song', 'artist',
                                                             'sessionId', 'location', 'userAgent')
    actions_df.show(10)

    # extract columns for users table  user_id, first_name, last_name, gender, level  
    users_table = df.selectExpr('userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level').dropDuplicates()
    print('First 10 Rows of users table')
    users_table.show(10)
    
    users_table.createOrReplaceTempView('users')
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users/users.parquet'), 'overwrite')


    # create timestamp column from original timestamp column
    ts_data = actions_df.selectExpr('ts').show(10)
    #function for timestamp
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    actions_df = actions_df.withColumn('timestamp', get_timestamp(actions_df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    actions_df = actions_df.withColumn('datetime', get_datetime(actions_df.ts))
    
    print('converted ts')
    actions_df.show(10)
    
    # extract columns to create time table 
    # time table: start_time, hour, day, week, month, year, weekday
    time_table = actions_df.selectExpr('datetime')\
                           .withColumn('start_time', actions_df.datetime) \
                           .withColumn('hour', hour('datetime')) \
                           .withColumn('day', dayofmonth('datetime')) \
                           .withColumn('week', weekofyear('datetime')) \
                           .withColumn('month', month('datetime')) \
                           .withColumn('year', year('datetime')) \
                           .withColumn('weekday', dayofweek('datetime')) \
                           .dropDuplicates()
        
    time_table.show(10)
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month') \
                    .parquet(os.path.join(output_data,
                                          'time/time.parquet'), 'overwrite')

    # read in song data to use for songplays table
    song_df = spark.read.json("./data/song_data/A/A/A/*.json")

    # extract columns from joined song and log datasets to create songplays table
    #Songplays table: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    actions_df = actions_df.alias('log_df')
    print('schema of the action table')
    actions_df.printSchema()
    
    song_df = song_df.alias('song_df')
    print('schema of the song_df')
    song_df.printSchema()
    
    #join filtered table with song table
    joined_df = actions_df.join(song_df, col('log_df.artist') == col('song_df.artist_name'), 'inner')
    
    songplays_table = joined_df.select(
                                       col('log_df.datetime').alias('start_time'),
                                       col('log_df.userId').alias('user_id'),
                                       col('log_df.level').alias('level'),
                                       col('song_df.song_id').alias('song_id'),
                                       col('song_df.artist_id').alias('artist_id'),
                                       col('log_df.sessionId').alias('session_id'),
                                       col('log_df.location').alias('location'), 
                                       col('log_df.userAgent').alias('user_agent'),
                                       year('log_df.datetime').alias('year'),
                                       month('log_df.datetime').alias('month')) \
                               .withColumn('songplay_id', monotonically_increasing_id())
 
    print('songplays table')
    songplays_table.show(10)
    
    songplays_table.createOrReplaceTempView('songplays')
    # write songplays table to parquet files partitioned by year and month
    time_table = time_table.alias('timetable')

    songplays_table.write.partitionBy('year', 'month')\
                         .parquet(os.path.join(output_data,
                                 'songplays/songplays.parquet'),'overwrite')

def main():
    """
    The following steps will be execudet:
    --> 1.) Get or create a spark session.
    --> 1.) Read the song and log data from s3 Bucket.
    --> 2.) take the needed data and transform them to tables and write them to parquet files.
    --> 3.) Load the parquet files on s3.
    """
    
    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    #output_data = "./data/result"  # to work in the workbook
    output_data = "s3a://result-data/"
    
    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
