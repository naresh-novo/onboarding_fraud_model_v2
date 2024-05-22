#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: naresh
"""
import numpy as np
import pandas as pd

from colorama import Fore



def Extract(lst):
    return list(list(zip(*lst)))[1]


def fraud_rate(df, target):
    test1 = pd.DataFrame(df[df[target]==1]['binned'].value_counts()).reset_index(drop=False)
    test0 = pd.DataFrame(df[df[target]==0]['binned'].value_counts()).reset_index(drop=False)
    test = pd.merge(test0,test1, on='index', how='inner')
    test[['bin','non_fraud_cnt','fraud_cnt']] = test[['index','binned_x','binned_y']]
    test.drop(columns=['index','binned_x','binned_y'],inplace=True)
    test = test.sort_values('bin').reset_index(drop=True)
    test['fraud_rate%'] = np.round(test['fraud_cnt']*100/(test['fraud_cnt']+test['non_fraud_cnt']),2)
    test['vol_cum%'] = np.int64((test['fraud_cnt'].cumsum() + test['non_fraud_cnt'].cumsum())*100 / (test['fraud_cnt'].sum() + test['non_fraud_cnt'].sum()))
    test['fraud_rate_cum%'] = np.round(test['fraud_cnt'].cumsum()*100/(test['fraud_cnt'].sum()+test['non_fraud_cnt'].sum()),2)
    test['%_of_fraud_cum'] = np.int64(test['fraud_rate_cum%']*100/test.loc[test.shape[0]-1,'fraud_rate_cum%'])

    test.rename(columns={'fraud_rate%':'bin_fraud_rate%', 'fraud_rate_cum%':'vol_fraud_rate_cum%'} ,inplace=True)
    return test


def ks_decile(data=None,target=None, prob=None, decile=True, n_decile=10, bins=None):
    if decile:
        data['bucket'] = pd.qcut(data[prob], n_decile, duplicates='drop')
    elif bins != None and len(bins) >=2:
        data['bucket'] = pd.cut(data[prob], bins=bins)
    data['target0'] = 1 - data[target]
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['bin'] =grouped.min()['bucket']
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100
    kstable['bin_event_rate'] = (np.round(kstable['events']/(
        kstable['events']+kstable['nonevents']),4)).apply('{0:.2%}'.format)
    kstable['cum_vol'] = (np.round((kstable['events']+kstable['nonevents']).cumsum()/(
        kstable['events']+kstable['nonevents']).sum(),4)).apply('{0:.1%}'.format)

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    if decile:
        kstable.index = range(1,kstable.shape[0]+1)
    else:
        kstable.index = range(1,kstable.shape[0]+1)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 20)
#     print(kstable)
    
    #Display KS
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
                
    return kstable



def score_bins(data=None,target=None, prob=None, decile=True, n_decile=10, bins=None):
    
    if decile:
        data['bucket'] = pd.qcut(data[prob], n_decile, duplicates='drop')
    elif bins != None and len(bins) >=2:
        data['bucket'] = pd.cut(data[prob], bins=bins)
    data['target'] = 1
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['bin'] =grouped.min()['bucket']
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['cum_vol'] = (np.round((kstable['target']).cumsum()/(
        kstable['target']).sum(),4)).apply('{0:.1%}'.format)
    
    if decile:
        kstable.index = range(1,kstable.shape[0]+1)
    else:
        kstable.index = range(1,kstable.shape[0]+1)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 20)
    
    return kstable