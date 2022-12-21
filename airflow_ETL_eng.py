from datetime import datetime, timedelta
import pandas as pd
import pandahouse as ph


from airflow.decorators import dag, task
from airflow.operators.python import get_current_context


# connection details are removed from the code
connection = {'host': 'https://clickhouse.lab.karpov.courses',
                      'database':'simulator_20221120',
                      'user':'', 
                      'password':''
                     }

connection2 = {'host': 'https://clickhouse.lab.karpov.courses',
                      'database':'test',
                      'user':'', 
                      'password':''
                     }



# default arguments
default_args = {
    'owner': 'n-vasilenko-13',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 12, 7),
}

# schedule interval for DAGs
schedule_interval = '0 23 * * *'

#DAG for daily auto-completion of a new table with the calculated data for the previous day. (extracting data from two tables, transforming and loading data into a new table)
@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def vasilenko_dag():

    @task()
    def extract_feed():
        query = """ SELECT toDate(time) as event_date, user_id, sum(action = 'like') as likes, sum(action = 'view') as views, gender, age, os
                FROM simulator_20221120.feed_actions 
                WHERE toDate(time) = today() - 1
                GROUP BY event_date, user_id, gender, age, os
                """
        df_feed = ph.read_clickhouse(query, connection=connection)

        return df_feed

    
    @task()
    def extract_messages():
        query1 = """ with messages as (
                SELECT toDate(time) as event_date, user_id, count(reciever_id) as messages_sent, COUNT(DISTINCT reciever_id) as users_sent, gender, age, os
                FROM simulator_20221120.message_actions  
                WHERE toDate(time) = today() - 1
                GROUP BY event_date, user_id, gender, age, os
                ),
                received_users as (
                SELECT toDate(time) as event_date, reciever_id, COUNT(DISTINCT user_id) as users_received, COUNT(user_id) as messages_received
                FROM simulator_20221120.message_actions  
                WHERE toDate(time) = today() - 1
                GROUP BY event_date, reciever_id
                )
                SELECT * FROM messages 
                JOIN received_users ON messages.user_id = received_users.reciever_id AND messages.event_date = received_users.event_date
                """
        df_messages = ph.read_clickhouse(query1, connection=connection)

        return df_messages    
    
    @task()
    def transfrom_source(df_feed, df_messages):
        df_united = df_feed.merge(df_messages, how='left', on=['event_date','user_id', 'gender','age','os']).drop(columns=['received_users.event_date', 'reciever_id'])
        
        return df_united

    @task()
    def transfrom_gender(df_united):
        df_gender = df_united.groupby(['event_date','gender'], as_index = False).agg('sum')
        df_gender['dimension'] = 'gender'
        df_gender = df_gender.rename(columns={'gender': 'dimension_value'})
        df_gender = df_gender[['event_date', 'dimension','dimension_value', 'views', 'likes', 'messages_received' ,'messages_sent', 'users_received', 'users_sent']]
        df_gender[['messages_received' ,'messages_sent', 'users_received', 'users_sent']] = df_gender[['messages_received' ,'messages_sent', 'users_received', 'users_sent']].astype('int64')
        
        return df_gender
    
    @task()
    def transfrom_age(df_united):
        df_age = df_united.groupby(['event_date','age'], as_index = False).agg('sum')
        df_age['dimension'] = 'age'
        df_age = df_age.rename(columns={'age': 'dimension_value'})
        df_age = df_age[['event_date', 'dimension','dimension_value', 'views', 'likes', 'messages_received' ,'messages_sent', 'users_received', 'users_sent']]
        df_age[['messages_received' ,'messages_sent', 'users_received', 'users_sent']] = df_age[['messages_received' ,'messages_sent', 'users_received', 'users_sent']].astype('int64')
        
        return df_age
    
    @task()
    def transfrom_os(df_united):
        df_os = df_united.groupby(['event_date','os'], as_index = False).agg('sum')
        df_os['dimension'] = 'os'
        df_os = df_os.rename(columns={'os': 'dimension_value'})
        df_os = df_os[['event_date', 'dimension','dimension_value', 'views', 'likes', 'messages_received' ,'messages_sent', 'users_received', 'users_sent']]
        df_os[['messages_received' ,'messages_sent', 'users_received', 'users_sent']] = df_os[['messages_received' ,'messages_sent', 'users_received', 'users_sent']].astype('int64')
        
        return df_os 
    
    
    @task()
    def load(df_gender, df_age, df_os):
        
        query2 = '''CREATE TABLE IF NOT EXISTS test.vasilenkodata
        (
            event_date Date,
            dimension String,
            dimension_value String,
            views Int64,
            likes Int64,
            messages_received Int64,
            messages_sent Int64,
            users_received Int64,
            users_sent Int64
            ) ENGINE = Log()'''
        ph.execute(query2, connection=connection2)
        ph.to_clickhouse(df_gender, table='vasilenkodata', connection=connection2, index = False)
        ph.to_clickhouse(df_age, table='vasilenkodata', connection=connection2, index = False)
        ph.to_clickhouse(df_os, table='vasilenkodata', connection=connection2, index = False)
        
    df_feed = extract_feed()
    df_messages = extract_messages()
    df_united = transfrom_source(df_feed, df_messages)
    df_gender = transfrom_gender(df_united)
    df_age = transfrom_age(df_united)
    df_os = transfrom_os(df_united)  
    load(df_gender, df_age, df_os)
    
vasilenko_dag = vasilenko_dag()
