# %%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# %%
df = pd.read_csv('../data/sample_submission.csv')
df.head()
df.tail()

# %%
test_station = sorted(set([df["id"][i].split("_")[1] for i in range(len(df))]))

# %%
def get_data(start_date, end_date):
    # get data from "release"
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    # Iterate through the range of dates
    data_path = "../..//html.2023.final.data/release/"
    test_data_dic = {}
    current_date = start_date
    while current_date <= end_date:
        folder_path = os.path.join(data_path, current_date.strftime("%Y%m%d"))
        for station in test_station:
            file_path = os.path.join(folder_path, station) + ".json"
            with open(file_path, 'r') as f:
                file = json.load(f)
                current_time = datetime.strptime("00:00", "%H:%M")
                end_time = datetime.strptime("23:59", "%H:%M")
                while current_time <= end_time:
                    time = current_time
                    while file[time.strftime("%H:%M")] == {}:
                        time += timedelta(minutes=1)
                    key = current_date.strftime("%Y%m%d") + "_" + station + "_" + current_time.strftime("%H:%M")
                    value = file[time.strftime("%H:%M")]
                    test_data_dic[key] = value
                    current_time += timedelta(minutes=20)
        current_date += timedelta(days=1)
    return test_data_dic

# %%
def validation(start_date, end_date, submission):
    ans = get_data(start_date, end_date)
    if set(ans.keys()) == set(submission.keys()):
        raise ValueError("Dictionaries do not have the same keys.")
    
    E_val = 0
    try: 
        for key in ans.keys():
            b_it = ans[key]["sbi"]
            b_it_pred = submission[key]
            s_i = ans[key]["tot"]
            E_val += 3 * abs((b_it -  b_it_pred) / s_i) * (abs(b_it / s_i - 1 / 3) + abs(b_it / s_i - 2 / 3))
    except: 
        raise ValueError("Inappropriate value detected.")
        
    return E_val


