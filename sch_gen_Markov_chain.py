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

#%%
def make_P_matrix(T_dict):
    P_matrix = np.zeros([time_lenth, len(state_l), len(state_l)])
    P_matrix[:, 0, 1] = T_dict['(H->W)']
    P_matrix[:, 0, 2] = 0
    P_matrix[:, 0, 3] = 0
    P_matrix[:, 0, 0] = 1 - T_dict['(H->W)']

    P_matrix[:, 1, 0] = T_dict['(W->H)']
    P_matrix[:, 1, 2] = T_dict['(W->L)']
    P_matrix[:, 1, 3] = T_dict['(W->R)']
    P_matrix[:, 1, 1] = 1 - (T_dict['(W->H)'] + T_dict['(W->L)'] + T_dict['(W->R)'])

    P_matrix[:, 2, 0] = 0
    P_matrix[:, 2, 1] = T_dict['(L->W)']
    P_matrix[:, 2, 2] = 1 - T_dict['(L->W)']
    P_matrix[:, 2, 3] = 0

    P_matrix[:, 3, 0] = 0
    P_matrix[:, 3, 1] = T_dict['(R->W)']
    P_matrix[:, 3, 2] = 0
    P_matrix[:, 3, 3] = 1 - T_dict['(R->W)']
    return P_matrix
def Transition_P(p):
    T00, T01 = 1, np.zeros(len(p))
    for i in range(len(p)):
        if 0.01 > (1 - p[i]):
            T01[i] = np.nan
        elif i == len(p) - 1:
            T01[i] = (1 - p[i]) / (1 - p[i])
        else:
            T01[i] = (p[i + 1] - p[i]) / (1 - p[i])
    idx_end = (p <0.99).sum()
    T01[idx_end:] = 0
    return np.nan_to_num(T01, nan=0) # T01

def Weibull_cdf(k, l, u, t):
    p_w = np.zeros(len(t))
    p_w[int(u):] = 1-np.exp(-np.power((t[int(u):]-int(u))/l, k))
    return p_w
def Weibull_pdf(k, l, u, t):
    p_w = k/l*((t-u)/l**(k-1))*np.exp(-np.power((t-u)/l, k))
    return p_w
def exponential_cdf(lambd,u, t):
    p = 1 - np.exp(-1/lambd*(t-u))
    p[:int(u)] = 0
    return p

#%% expected sch result view
def probability_view(df_state):
    fig1, ax = plt.subplots(3, 2)
    ax = ax.flatten()
    for k in keys_T:
        ax[0].plot(df_state[k + '_cdf'], label=k)
        ax[0].axvline(x=df_state.index[u_dict[k]], color='grey', linestyle='--')
        ax[0].set(ylabel='CDF',ylim=(0,1.2))
        ax[1].bar(df_state.index, df_state[k + '_T'], label=k, width=0.005)
        ax[1].set(ylabel='Transition Probability',ylim=(0,1.2))
        ax[1].axvline(x=df_state.index[u_dict[k]], color='grey', linestyle='--')
    for k in keys_D:
        ax[2].plot(df_state[k + '_cdf'], label=k)
        ax[2].set(ylabel='CDF',ylim=(0,1.2))
        ax[3].plot(df_state.index, df_state[k + '_T'], label=k)
        ax[3].set(ylabel='Transition Probability',ylim=(0,1.2))
    for i,k in enumerate(state_dict.keys()):
        ax[4].plot(df_state[k], linewidth=1, label=k)
        ax[4].set(ylabel='State',ylim=(0,1.2))
    ax[5].plot(df_state['Work'], linewidth=1, label='Work', color='r')
    ax[5].set(ylabel='State',ylim=(0,1.2))
    for a in ax:
        a.legend()
        a.xaxis.set_major_formatter(dateFmt)
def result_view(data):
    fig, ax = plt.subplots(1,1)
    ax.plot(data, alpha=0.5, label='Expected Occupancy')
    ax.xaxis.set_major_formatter(dateFmt)
    ax.set(ylim=(0,1.2))
    ax.legend()

#%% Weibull distribution & view transition probability
t = np.linspace(0,144,144+1)
time_lenth = len(t)
time_index = pd.date_range('2021-10-1 00:00:00', '2021-10-2 00:00:00', freq='10min')
state_l = ['Home', 'Work', 'Lunch', 'Rest']
state_dict = {'Home':np.array([1,0,0,0]),
              'Work':np.array([0,1,0,0]),
              'Lunch':np.array([0,0,1,0]),
              'Rest':np.array([0,0,0,1])}

keys_T = ['(H->W)', '(W->L)','(L->W)','(W->H)']
k_dict = {'(H->W)':2,'(W->L)':3,'(L->W)':2, '(W->H)':1.5}
l_dict = {'(H->W)':4,'(W->L)':4,'(L->W)':4, '(W->H)':4}
u_dict = {'(H->W)':6*8,'(W->L)':6*11.5, '(L->W)':6*12.5, '(W->H)':6*17.5}
p_dict = {j:Weibull_cdf(k_dict[j], l_dict[j], u_dict[j], t) for j in keys_T}
T_dict = {j:Transition_P(p_dict[j]) for j in keys_T}
T_dict['(W->H)'][T_dict['(W->H)'].argmax()+1:] = T_dict['(W->H)'].max()

keys_D = ['(W->R)', '(R->W)']
lam_dict = {'(W->R)':10, '(R->W)':2}
u_dict_D = {'(W->R)':0, '(R->W)':0}

for i in keys_D:
    p_dict[i] = exponential_cdf(lam_dict[i], u_dict_D[i], t)
    T_dict[i] = np.ones(time_lenth)*Transition_P(p_dict[i]).max()
keys_total = keys_T+ keys_D

change_state_dict = {'(H->W)': state_dict['Work'] - state_dict['Home'],
                     '(W->L)': state_dict['Lunch'] - state_dict['Work'],
                     '(W->H)': state_dict['Home'] - state_dict['Work'],
                     '(L->W)': state_dict['Work'] - state_dict['Lunch'],
                     '(W->R)': state_dict['Rest'] - state_dict['Work'],
                     '(R->W)': state_dict['Work'] - state_dict['Rest']}

#%% calculation expected sch
P_matrix = make_P_matrix(T_dict)
P_init = [1,0,0,0]
state_p = P_init
State_person_array, state_p_array = np.zeros([time_lenth, len(P_init)]), np.zeros([time_lenth, len(P_init)])
for t_i in range(time_lenth):
    State_person_array[t_i] = state_p
    state_p = np.dot(state_p, P_matrix[t_i])
df_state = pd.DataFrame()
for i,k in enumerate(keys_total):
    df_state[k+'_cdf'] = p_dict[k]
    df_state[k+'_T'] = T_dict[k]
for i,k in enumerate(state_dict.keys()):
    df_state[k] = State_person_array[:,i]
df_state.index = time_index


#%% calculation expected sch
def gen_sch(P_matrix):
    p_u = np.zeros([time_lenth, len(P_init)])
    State_array = np.zeros([time_lenth, len(P_init)])
    p_absence = np.random.normal(loc=0.16, scale=0.032) # 결근 확률: 근무일 대비 평균 휴가일(20), 출장(20.4일) 20+20.4/247
    state_abs = np.random.choice(['attend', 'absence'], p=[1-p_absence,p_absence])
    state_ = [1,0,0,0]# P_init_dict[state_abs]
    for t_i in range(time_lenth):
        if state_abs == 'absence': # 결근자 마코프 미적용
            State_array[:, 0] = 1
            # print('absence!')
            break
        State_array[t_i] = state_
        state_p = np.dot(state_, P_matrix[t_i])
        state_ =  state_dict[np.random.choice(state_l, p=state_p)]
        p_u[t_i] = (state_ - State_array[t_i]) # 상태전환점
        if np.all(p_u[t_i] == change_state_dict['(W->L)']):
            # print(df_state.index[t_i].time(),'(W->L)')
            p_LW = Weibull_cdf(k_dict['(L->W)'], l_dict['(L->W)'], t_i, t) # 상태전환점 기준으로 복귀시간 확률 결정
            T_dict['(L->W)'] = Transition_P(p_LW)
            P_matrix = make_P_matrix(T_dict)
    return State_array

df_work = pd.DataFrame()

def main():
    iter_count = 300
    for i in tqdm(range(iter_count)):
        State_array = gen_sch(P_matrix)
        df_work['sch_' + str(i)] = State_array[:, 1]
    df_work.index = time_index
#%%
if __name__ == '__main__':
    main()