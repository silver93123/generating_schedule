import numpy as np
import scipy as sc
from tqdm import tqdm
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import scipy.stats as ss
import matplotlib.dates as mdates
import pythermalcomfort as pt
dateFmt = mdates.DateFormatter('%H:%M')
#%%
def cal_cooling_setting_temp(met): # cooling
    temp_range = np.linspace(30, 16, 600)
    pmv = np.nan
    setting_temp = np.nan
    for i_t,t in enumerate(temp_range):
        pmv = pt.pmv(t, t, 0.1, 50, met=met, clo=0.9)
        pmv_m = np.random.uniform(0, 0.5)
        if pmv <= pmv_m:
            setting_temp = t
            break
    return setting_temp, pmv
def cal_setting_temp(met, mode, clo): # cooling
    pmv, setting_temp = np.nan, np.nan
    if mode =='Heating':
        temp_range = np.linspace(16, 30, 600)
        pmv_m = np.random.uniform(-0.5, 0)
        for i_t,t in enumerate(temp_range):
            pmv = pt.pmv(t, t, 0.1, 50, met=met, clo=clo)
            if pmv >= pmv_m:
                setting_temp = t
                break
    else:
        temp_range = np.linspace(30, 16, 600)
        pmv_m = np.random.uniform(0, 0.5)
        for i_t, t in enumerate(temp_range):
            pmv = pt.pmv(t, t, 0.1, 50, met=met, clo=0.9)
            if pmv <= pmv_m:
                setting_temp = t
                break
    return round(setting_temp,3), pmv
def cal_met_ISO8996(sex, weight, height, age, HR_m):
    A_du = pt.body_surface_area(weight, height)
    if sex == 'Men':
        M_0 = 60*A_du
        W_bl = (1.08 - weight / (80 * (height ** 2))) * weight
        MWC = (19.45 - 0.133*age)*W_bl
        HR_0, HR_m = 60,HR_m
    else:
        M_0 = 55*A_du
        W_bl = (0.86 - weight / (107.5 * (height ** 2))) * weight
        MWC = (17.51 - 0.15*age)*W_bl
        HR_0, HR_m = 60,HR_m
    HR_max = 208-0.7*age
    RM = (HR_max - HR_0)/(MWC - M_0)
    M_m = M_0 + (HR_m - HR_0)/RM # W
    M = M_m/A_du # W/m2
    met =M/58.18
    return met
def gen_feature_human():
    sex = np.random.choice([1, 2], p=[0.55, 0.45])
    if sex == 1: # Man
        age = round(np.random.normal(42, 7))
        height = round(np.random.normal(1.74, 0.04), 3)
        BMI = round(np.random.normal(30.0, 0.8), 3)
        weight = abs(round((height ** 2) * BMI, 2))
        sex = 'Men'
    else: # Woman
        age = round(np.random.normal(37,7))
        height = round(np.random.normal(1.58,0.04),3)
        BMI = round(np.random.normal(29.,0.6),3)
        weight = abs(round((height**2)*BMI,2))
        sex = 'Woman'
    return {'age': age, 'height':height, 'BMI':BMI, 'weight':weight, 'sex':sex }

#%% 설정온도 시뮬레이션
iter_ = 100
clo_l = [1.0,0.8]
df_data = pd.DataFrame()
def main():
    for i_m,m in enumerate(['Heating', 'Cooling']):
        df_ = pd.DataFrame(columns=['age', 'BMI', 'weight', 'height', 'met', 'pmv','temp_set', 'sex', 'mode'], index=range(iter_))
        df_['mode'] = m
        for i in tqdm(range(len(df_))):
            ocp = gen_feature_human()
            HR_m = round(np.random.normal(67,1))
            met = cal_met_ISO8996(2 ,ocp['weight'],ocp['height'], ocp['age'], HR_m)
            temp_set, pmv = cal_setting_temp(met,m,clo_l[i_m])
            df_.iloc[i,:8] = [ocp['age'], ocp['BMI'], ocp['weight'], ocp['height'], met, pmv, temp_set, ocp['sex']]
        df_data = pd.concat([df_data, df_])
    df_data = df_data.reset_index(drop=True)
    df_data = df_data.astype('f', errors='ignore')
