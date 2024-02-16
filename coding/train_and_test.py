# %%
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from libsvm.svmutil import *
from liblinear.liblinearutil import *
import matplotlib.pyplot as plt


# %%
# load demographic
with open("../data/new_demographic.json", 'r')as f:
    demo = json.load(f)
# load all stations
all_stations = sorted([f[:-5] for f in os.listdir("../../html.2023.final.data/release/20231002")])
sample_sub = pd.read_csv('../data/sample_submission.csv')
test_stations = sorted(set([sample_sub["id"][i].split("_")[1] for i in range(len(sample_sub))]))
test_stations_idx = [i for i, station in enumerate(all_stations) if station in test_stations]
problem_date = {datetime(2023, 10, 12, 0, 0), datetime(2023, 10, 13, 0, 0), datetime(2023, 10, 14, 0, 0), datetime(2023, 10, 15, 0, 0),
                datetime(2023, 10, 21, 0, 0), datetime(2023, 10, 22, 0, 0), datetime(2023, 10, 23, 0, 0), datetime(2023, 10, 24, 0, 0),
                datetime(2023, 11, 30, 0, 0), datetime(2023, 12, 1, 0, 0)}
# %%
def get_train_data(start_date, end_date, selected_stations):
    df = [[] for _ in range(len(selected_stations))]
    columns = []
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    

    current_date = start_date
    while current_date <= end_date:
        if current_date not in problem_date:
            folder_path = f'../../html.2023.final.data/release/{current_date.strftime("%Y%m%d")}/'
            print(current_date)
            for i, station in enumerate(selected_stations):
                try:
                    with open(os.path.join(folder_path, station) + ".json", 'r')as f:
                        data = json.load(f)
                        for key, value in data.items():
                            time = key
                            # data filling
                            if value == {}:
                                timestamp = datetime.strptime(key, "%H:%M") + timedelta(minutes=1)
                                while data[timestamp.strftime("%H:%M")] == {}:
                                    timestamp += timedelta(minutes=1)
                                value = data[timestamp.strftime("%H:%M")]
                            df[i].append(value)
                            if i == 0:
                                columns.append(f"{current_date.strftime('%Y%m%d')}_{time}")
                except:
                    break
        current_date += timedelta(days=1)

    return pd.DataFrame(df, index=selected_stations, columns=columns).T

# %%
def get_all_tot(date):
    dic_tot = {}
    folder_path = f'../../html.2023.final.data/release/{date}/'
    for station in all_stations:
        with open(os.path.join(folder_path, station) + ".json", 'r')as f:
            data = json.load(f)
            for key, value in data.items():
                # data filling
                if value == {}:
                    timestamp = datetime.strptime(key, "%H:%M") + timedelta(minutes=1)
                    while data[timestamp.strftime("%H:%M")] == {}:
                        timestamp += timedelta(minutes=1)
                    value = data[timestamp.strftime("%H:%M")]
                dic_tot[station] = value['tot']
                break
    return dic_tot

def old_feature_selection(df, data_num = 0, interval = 20, min_per_data = 20):
    stations = df.columns
    time_period = df.index
    all_date = sorted(set([time[:-6] for time in time_period]))
    start_date = datetime.strptime(all_date[0], "%Y%m%d")
    
    x_train = [[] for _ in range(len(stations))]
    
    for i, station in enumerate(stations):
        station_data = df[station].values
        station_demo = demo[station]
        indices = (np.arange(data_num) * interval).tolist()
        ts = station_demo["ts"]
        ts_id = station_demo["ts_id"]
        # weather dataframe list for this station
        df_weather = []
        for date in all_date:
            df_w = pd.read_csv(f'../data/weather_data/{ts}/{ts_id}-{date}.csv')
            temp_list = df_w['氣溫(℃)']
            prcp_list = df_w['降水量(mm)']
            ws_list = df_w['風速(m/s)']
            rh_list = df_w['相對溼度(%)']
            df_weather.append({"temp": temp_list, "prcp": prcp_list, "rh": rh_list, 'ws': ws_list,
                "temp_avg": round(np.average([float(value) for value in df_w['氣溫(℃)'] if value.replace(".", "").isdigit()]), 1)
})
        # initialize weekday
        weekday = start_date.weekday()
        
        for j in range(interval * data_num, len(station_data), min_per_data):
            day = j // 1440
            hour = (j % 1440 ) // 60
            minute = (j % 1440 ) % 60
            
            # get date in datetime
            date = start_date + timedelta(days=day, hours=hour, minutes=minute)
            
            # get lng, lat from demographic
            lng, lat = station_demo["lng"], station_demo["lat"]
            
            # weekday
            is_day = (weekday + day) % 7
            weekday_list = [0 for _ in range(7)]
            weekday_list[is_day] = 1
            
            # read weather csv
            temp = df_weather[day]['temp'][hour + 1]
            prcp = df_weather[day]['prcp'][hour + 1]
            rh = df_weather[day]['rh'][hour + 1]
            ws = df_weather[day]['ws'][hour + 1]
            if prcp == 'T' or prcp == '&':
                prcp = 0
            elif float(prcp) > 0:
                prcp = 1
            if temp == '--':
                temp = df_weather[day]['temp_avg']
            if rh == '--':
                rh = 0
            if ws == '--':
                ws = 0
            
            x_train[i].append([station, date, lng, lat, hour, minute, station_data[j]["sbi"], station_data[j]["tot"]] + weekday_list + [temp, prcp, rh, ws])
    
    return np.array(x_train), ['station_id', 'date', 'lng', 'lat', 'hour', 'minute', 'sbi', 'tot', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun', 'temp', 'prcp', 'rh', 'ws']

def feature_selection(df, data_num = 0, interval = 20, min_per_data = 20):
    stations = df.columns
    time_period = df.index
    all_date = sorted(set([time[:-6] for time in time_period]))
    start_date = datetime.strptime(all_date[0], "%Y%m%d")
    
    x_train = [[] for _ in range(len(stations))]
    
    for i, station in enumerate(stations):
        station_data = df[station].values
        station_demo = demo[station]
        # initialize weekday
        weekday = start_date.weekday()
        
        for j in range(interval * data_num, len(station_data), min_per_data):
            day = j // 1440
            hour = (j % 1440 ) // 60
            minute = (j % 1440 ) % 60
            
            # get date in datetime
            date = start_date + timedelta(days=day, hours=hour, minutes=minute)
            
            # get lng, lat from demographic
            lng, lat = station_demo["lng"], station_demo["lat"]
            
            # weekday
            is_day = (weekday + day) % 7
            weekday_list = [0 for _ in range(7)]
            weekday_list[is_day] = 1
            
            x_train[i].append([station, date, lng, lat, hour, minute, station_data[j]["sbi"], station_data[j]["tot"]] + weekday_list)
    
    return np.array(x_train), ['station_id', 'date', 'lng', 'lat', 'hour', 'minute', 'sbi', 'tot', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
# %%
def model_selection(x_train, y_train, model_name):
    if model_name == 'lin_reg':
        reg = LinearRegression().fit(x_train, y_train)
        return reg
    elif model_name == 'rf':
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1,oob_score=True)
                                            #  max_depth=19, min_samples_split=8, min_samples_leaf=3)
        rf_regressor.fit(x_train, y_train)
        return rf_regressor
    elif model_name == 'svr':
        prob = problem(list(y_train), list(x_train))
        params = " -s 11 -c 1000"
        svr = train(prob, params)
        return svr
    elif model_name == 'adaboost':
        base_estimator = DecisionTreeRegressor(max_depth=13)
        adaboost_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=150, learning_rate=0.1, random_state=42)
        adaboost_regressor.fit(x_train, y_train)
        return adaboost_regressor
    elif model_name == 'xgboost':
        xgb = XGBRegressor(max_depth=7, n_estimators=1000, learning_rate= 0.1)
        xgb.fit(x_train, y_train)
        return xgb
# %%
def select_data(df_feat, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")
    
    return df_feat[(df_feat['date'] >= start_date) & (df_feat['date'] < end_date + timedelta(days=1))].reset_index(drop = True)

# %%
def calculate_Ein(x_train, y_train, slot_num):
    pred = model_selection(x_train, y_train, 'lin_reg').predict(x_train)
    E_in = []
    for i in range(len(pred)):
        E_in.append(3 * abs((y_train[i] -  pred[i]) / slot_num[i]) * (abs(y_train[i] / slot_num[i] - 1 / 3) + abs(y_train[i] / slot_num[i] - 2 / 3)))
    return np.average(E_in)

# %%
def cross_validation(model_name, x, y, slot_num, n_splits = 5):
    x = np.vstack(x)
    y = y.flatten()
    slot_num = slot_num.flatten()
    
    kfold = KFold(n_splits=n_splits, shuffle=True)
    Average_Eval = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
        x_train, x_test = [x[i] for i in train_idx], [x[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
        slot_train, slot_test = [slot_num[i] for i in train_idx], [slot_num[i] for i in test_idx]

        # Replace this with your model fitting and evaluation
        print(f"Fold {fold_idx + 1}:")
        model = model_selection(x_train, y_train, model_name)
        E_val = []
        pred = model.predict(x_test)
        for j in range(len(pred)):
            E_val.append(3 * abs((y_test[j] -  pred[j]) / slot_test[j]) * (abs(y_test[j] / slot_test[j] - 1 / 3) + abs(y_test[j] / slot_test[j] - 2 / 3)))
        print(f"Validation for Fold {fold_idx + 1} = {np.average(E_val)}")
        Average_Eval.append(np.average(E_val))
    print(f"\nOverall Performance = {np.average(Average_Eval)}")   
    return np.average(Average_Eval)

# %%
def old_sequence_validation(model_name, x, y, slot_num, time_span, train_days, test_days, history = False):
    split_point = -test_days * 72
    Average_Eval = []
    for i in range(time_span - (train_days + test_days) + 1):
        
        x_use = x[:, i * 72: (i + train_days + test_days) * 72]
        y_use = y[:, i * 72: (i + train_days + test_days) * 72]
        slot_use = slot_num[:, i * 72: (i + train_days + test_days) * 72]
        x_train = x_use[:, :split_point]
        y_train = y_use[:, :split_point]
        # slot_train = (slot_use[:, :split_point]).flatten()
        x_test = x_use[:, split_point:]
        y_test = y_use[:, split_point:]
        slot_test = slot_use[:, split_point:]   
        
        print(len(np.vstack(x_train)), len(y_train.flatten()))
        E_val = []
        model = model_selection(np.vstack(x_train), y_train.flatten(), model_name)
            
        # using history
        if history:
            for j in range(len(x_train)):
                pred = [y_train[j][-1]]
                for k in range(len(x_test[j])):
                    cur = np.concatenate([x_test[j][k][:-1], pred])
                    pred = model.predict([cur])
                    E_val.append(3 * abs((y_test[j][k] -  pred[0]) / slot_test[j][k]) * (abs(y_test[j][k] / slot_test[j][k] - 1 / 3) + abs(y_test[j][k] / slot_test[j][k] - 2 / 3)))
        
        # not using history
        else:
            x_test = np.vstack(x_test)
            y_test = y_test.flatten()
            slot_test = slot_test.flatten()
            pred = model.predict(x_test)
            for j in range(len(pred)):
                E_val.append(3 * abs((y_test[j] -  pred[j]) / slot_test[j]) * (abs(y_test[j] / slot_test[j] - 1 / 3) + abs(y_test[j] / slot_test[j] - 2 / 3)))
        
        print(f"Validation for Fold {i} = {np.average(E_val)}")
        Average_Eval.append(np.average(E_val))
        # return E_val
    print(f"\nOverall Performance = {np.average(Average_Eval)}")   
    return np.average(Average_Eval)

def sequence_validation(model_name, df_feat, train_days, test_days, selected_feat):
    time_span = (list(df_feat['date'])[-1] - list(df_feat['date'])[0]).days + 1
    Average_Eval = []
    for i in range(time_span - train_days - test_days + 1):
        train_data = df_feat[(df_feat['date'] >= (df_feat['date'][0] + timedelta(days=i))) & (df_feat['date'] < df_feat['date'][0] + timedelta(days=train_days + i))]
        test_data = df_feat[(df_feat['date'] >= df_feat['date'][0] + timedelta(days=train_days + i)) & (df_feat['date'] < df_feat['date'][0] + timedelta(days=train_days + test_days + i))]
        
        print(f"Train Period:{list(train_data['date'])[0]} to {list(train_data['date'])[-1]}")
        print(f"Test Period:{list(test_data['date'])[0]} to {list(test_data['date'])[-1]}")
        
        x_train, y_train = train_data[selected_feat], train_data['sbi']
        x_test, y_test, slot_test = test_data[selected_feat], list(test_data['sbi']), list(test_data['tot'])

        E_val = []
        model = model_selection(x_train.values, y_train.values, model_name)
        if model_name == "svr":
            pred, p_acc, _ =  predict([1000 for _ in range(len(x_test))], list(x_test.values), model)
        else:
            pred = model.predict(x_test.values)
        for j in range(len(pred)):
            real_pred = min(max(0, pred[j]), slot_test[j])
            score = 3 * abs((y_test[j] -  real_pred) / slot_test[j]) * (abs(y_test[j] / slot_test[j] - 1 / 3) + abs(y_test[j] / slot_test[j] - 2 / 3))
            E_val.append(score)
        print(f"Validation for Fold {i} = {np.average(E_val)}\n")
        Average_Eval.append(np.average(E_val))
        # return E_val
    print(f"\nOverall Performance = {np.average(Average_Eval)}")   
    return np.average(Average_Eval)

# Testing
# %%
def get_test_data(dic_tot, start_datestr, end_datestr):
    x_test = [[] for _ in range(len(test_stations))]
    
    start_date = datetime.strptime(start_datestr, "%Y%m%d")
    end_date = datetime.strptime(end_datestr, "%Y%m%d")
    total_days = (end_date - start_date).days + 1
    all_date = [datetime.strftime(start_date + timedelta(days=i), "%Y%m%d") for i in range(total_days)]
    
    for i, station in enumerate(test_stations):
        station_demo = demo[station]
        tot = dic_tot[station]
        # initialize weekday
        weekday = start_date.weekday()
        
        for j in range(0, 1440 * total_days, 20):
            day = j // 1440
            hour = (j % 1440 ) // 60
            minute = (j % 1440 ) % 60
            
            # get date in datetime
            date = start_date + timedelta(days=day, hours=hour, minutes=minute)
            
            # get lng, lat from demographic
            lng, lat = station_demo["lng"], station_demo["lat"]
                    
            # weekday
            is_day = (weekday + day) % 7
            weekday_list = [0 for _ in range(7)]
            weekday_list[is_day] = 1
            
            x_test[i].append([station, date, lng, lat, hour, minute, tot] + weekday_list)
    
    return np.array(x_test), ['station_id', 'date', 'lng', 'lat', 'hour', 'minute', 'tot', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
# %%
def testing(model, x_test, tot):
    pred = model.predict(x_test)
    pred = model.predict(x_test)
    real_pred = [min(max(0, pred[i]), tot[i]) for i in range(len(pred))]
    return real_pred

# %%
def get_test_index(start_datestr, end_datestr):
    index = []
    start_date = datetime.strptime(start_datestr, "%Y%m%d")
    end_date = datetime.strptime(end_datestr, "%Y%m%d")
    
    for station in test_stations:
        current_date = start_date
        while current_date <= end_date:
            current_time = datetime.strptime("00:00", "%H:%M")
            end_time = datetime.strptime("23:59", "%H:%M")
            while current_time <= end_time:
                index.append(current_date.strftime("%Y%m%d") + "_" + station + "_" + current_time.strftime("%H:%M"))
                current_time += timedelta(minutes=20)
            current_date += timedelta(days=1)
    return index

# %%
def save_csv(index1, index2, pred):
    submission = pd.DataFrame([(index1 + index2), np.hstack((np.zeros(72 * len(test_stations) * 7), pred))], index=[['id', 'sbi']]).T
    submission.to_csv('../submission/3nd_submission.csv', index=False)


