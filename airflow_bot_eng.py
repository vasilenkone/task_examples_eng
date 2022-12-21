from datetime import datetime, timedelta
import telegram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pandas as pd
import pandahouse as ph

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

# connection details, chat id and bot token are removed from the code
connection = {'host': 'https://clickhouse.lab.karpov.courses',
                      'database':'simulator_20221120',
                      'user':'student', 
                      'password':'dpo_python_2020'
                     }

# default arguments
default_args = {
    'owner': 'n-vasilenko-13',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 12, 11),
}

# schedule interval for DAGs
schedule_interval = '0 11 * * *'



@dag(default_args=default_args, schedule_interval=schedule_interval, catchup=False)
def vasilenko_bot_dag():
    
    @task()
    def test_report(chat=None):

        chat_id = chat or -000000000 
        
        my_token = 'BOT_TOKEN'
        bot = telegram.Bot(token=my_token)
        
        query = ''' SELECT toDate(time) as date, count(DISTINCT user_id) as DAU, sum(action = 'like') as likes, sum(action = 'view') as views, likes/views as CTR
        FROM simulator_20221120.feed_actions WHERE toDate(time) >= today() - 7 AND toDate(time) < today() GROUP BY date'''
        
        data = ph.read_clickhouse(query, connection=connection)
        
        msg = (f"Data: {data['date'].astype('string')[0]}"
               f"\nMetrics:"
               f"\nDAU: {data['DAU'][data.index[-1]]}"
               f"\nlikes: {data['likes'][data.index[-1]]}"
               f"\nviews: {data['views'][data.index[-1]]}"
               f"\n CTR: {round(data['CTR'][data.index[-1]],3)}")
        
        msg2 = f"Plots for the period {data['date'].astype('string')[0]} - {data['date'].astype('string')[data.index[-1]]}"

        bot.sendMessage(chat_id=chat_id, text=msg)
        bot.sendMessage(chat_id=chat_id, text=msg2)
        
        sns.set_theme(style="darkgrid")
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(data['date'], data['DAU'])
        axs[0].set_ylabel('DAU')
        axs[0].grid(True)
        axs[1].plot(data['date'], data['likes'])
        axs[1].set_ylabel('likes')
        axs[1].grid(True)
        axs[2].plot(data['date'], data['views'])
        axs[2].set_ylabel('views')
        axs[2].grid(True)
        axs[3].plot(data['date'], data['CTR'])
        axs[3].set_ylabel('CTR')
        axs[3].grid(True)

        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout()

        plot_object = io.BytesIO()
        plt.savefig(plot_object)
        plot_object.seek(0)
        plot_object.name = 'metrics_plot.png'
        plt.close()
        bot.sendPhoto(chat_id=chat_id, photo=plot_object)

    test_report()
    
vasilenko_bot_dag = vasilenko_bot_dag()
