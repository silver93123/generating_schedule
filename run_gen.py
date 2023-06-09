import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.dates as mdates
dateFmt = mdates.DateFormatter('%H:%M')
import matplotlib
matplotlib.use('Qt5Agg')
import sch_gen_Markov_chain as sch_gen
import SetTemp_gen_ISO as SetTemp_gen

#%%
class occupancy:
    def __init__(self, spec):
        self.spec = spec
        self.met = np.nan
        self.set_cooling = np.nan
        self.set_heating = np.nan
        self.o_sch = np.nan
        self.heating_sch = np.nan
        self.cooling_sch = np.nan
        self.pmv_heating = np.nan
        self.pmv_cooling = np.nan

    def gen_settemp(self):
        HR_m = round(np.random.normal(67, 1))
        clo_ = round(np.random.normal(0.8, 0.1))
        self.met = SetTemp_gen.cal_met_ISO8996(self.spec['sex'], self.spec['weight'], self.spec['height'], self.spec['age'], HR_m)
        self.set_heating, self.pmv_heating = SetTemp_gen.cal_setting_temp(self.met, 'Heating', clo_+0.2)
        self.set_cooling, self.pmv_cooling = SetTemp_gen.cal_setting_temp(self.met, 'Cooling', clo_)

    def gen_sch(self):
        P_matrix = sch_gen.P_matrix
        self.o_sch = sch_gen.gen_sch(P_matrix)[:, 1]
        self.heating_sch = self.o_sch*self.heating_sch
        self.cooling_sch = self.o_sch*self.cooling_sch


#%%
time_index = pd.date_range('2021-10-1 00:00:00', '2021-10-2 00:00:00', freq='10min')
t = np.linspace(0,144,144+1)
time_lenth = len(t)
state_l = ['Home', 'Work', 'Lunch', 'Rest']

#%%
iter_count = 100
P_matrix = sch_gen.P_matrix
sch_array = np.zeros((iter_count, len(t)))

for i in tqdm(range(iter_count)):
    State_array = sch_gen.gen_sch(P_matrix)
    sch_array[i] = State_array[:, 1]

df_work = pd.DataFrame(data=sch_array.T, index=time_index)
#%%
sch_gen.result_view(df_work.mean(axis=1))

#%%

o_1 = occupancy(SetTemp_gen.gen_feature_human())
o_1.gen_sch()
o_1.gen_settemp()

sch_gen.result_view(o_1.o_sch)
print(o_1.set_cooling)
#%%
HR_m = round(np.random.normal(67, 1))
met = cal_met_ISO8996(ocp['sex'], ocp['weight'], ocp['height'], ocp['age'], HR_m)
temp_set, pmv = cal_setting_temp(met, m, clo_l[i_m])
