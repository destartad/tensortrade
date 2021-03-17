#%%from hmmlearn.hmm import GMMHMM,GaussianHMM
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot
import sys

def load_csv(filename):
    df = pd.read_csv(filename)
    #minute_EURUSD = minute_EURUSD.set_index('open_time')
    #minute_EURUSD.index = pd.to_datetime(minute_EURUSD.index)
    df['open_time']=pd.to_datetime(df['open_time'])
    df['weekofyear'] = df['open_time'].dt.weekofyear
    df['year'] = df['open_time'].dt.year

    df['month'] = df['open_time'].dt.month
    df['weekday'] = df['open_time'].dt.dayofweek
    df['day'] = df['open_time'].dt.day
    df['hour'] = df['open_time'].dt.hour
    df['minute'] = df['open_time'].dt.minute
    df.sort_values(by='open_time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    #df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S %p')
    #df=df.add_prefix("EUR:")
    return df

df = load_csv(sys.path[0] + "/tensortrade/data/EURUSD_1minute.csv")
"""
startdate = datetime.datetime.strptime('2020-07-01', '%Y-%m-%d')
enddate = datetime.datetime.strptime('2020-12-31','%Y-%m-%d')
#data = get_price('000300.XSHG',start_date=startdate, end_date=enddate,frequency='1d')
#print(data[0:6])
df = minute_EURUSD['open_time'][startdate:enddate]
"""

open=df['open'][5:]
close = df['close'][5:]
high = df['high'][5:]
low = df['low'][5:]

datelist = pd.to_datetime(close.index[:])
#logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))
#logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
logreturn = np.append(np.array([0]),np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))
logreturn5 = np.append(np.array([0,0,0,0,0]), np.log(np.array(close[5:]))-np.log(np.array(close[:-5])))
diffreturn = (np.log(np.array(high))-np.log(np.array(low)))
closeidx = close[:]
X = np.column_stack([open,close,high,low,logreturn,logreturn5,diffreturn])
#X = np.column_stack([diffreturn,logreturn,logreturn5])
print('open:'+str(len(open))+',close:'+str(len(close))+', high:'+str(len(high))+', low:'+str(len(low)))
print('X: '+str(len(X))+',    datelist: '+str(len(datelist))+', closeidx: '+str(len(closeidx)) )
print('diffreturn: '+str(len(diffreturn))+',    logreturn: '+str(len(logreturn))+', logreturn5: '+str(len(logreturn5)) )

#%%
from hmmlearn.hmm import GMMHMM,GaussianHMM

hmm = GaussianHMM(n_components = 6, covariance_type='diag',n_iter = 20000).fit(X)
latent_states_sequence = hmm.predict(X)
#print(hmm.startprob_)
#print(hmm.transmat_)
#print(hmm.mean_)
#print(hmm.covars_)
#latent_states_sequence=hmm.decode
print(len(latent_states_sequence))
print(latent_states_sequence)
print(hmm.score(X))

#%%
import matplotlib.pyplot as plt

sns.set_style('white')
plt.figure(figsize = (15, 8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    print(i,len(state),len(datelist[state]),len(closeidx[state]))
    plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
    plt.legend()
    plt.grid(1)

#%%
data = pd.DataFrame({'datelist':datelist,'logreturn':logreturn,'state':latent_states_sequence}).set_index('datelist')
plt.figure(figsize=(15,8))
for i in range(hmm.n_components):
    state = (latent_states_sequence == i)
    idx = np.append(0,state[:-1])
    data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 
    plt.plot(np.exp(data['state %d_return' %i].cumsum()),label = 'latent_state %d'%i)
    plt.legend()
    plt.grid(1)

# %%
