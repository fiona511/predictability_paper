#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:06:52 2024

@author: fionaspuler
"""


import xarray as xr
import numpy as np
import pandas as pd

def calculate_cluster_skill_score_probabilistic(cluster_number_pr, cluster_number_z500, 
                                                pr_cluster_labels, z500_cluster_probabilities):
    
    
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    conditional_probabilities = []
    for k1 in range(cluster_number_pr):
        for k2 in range(cluster_number_z500):

            p3 = z500_cluster_probabilities[:, k2].mean()
            p2 = (pr_cluster_labels==k1).mean()
            p1 = z500_cluster_probabilities[:, k2][pr_cluster_labels==k1].mean()

            p = p1*p2/p3

            conditional_probabilities.append(pd.DataFrame(data={
                            "cluster_pr": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [p]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)
    
    # issue forecast based on conditional probabilities
    forecast = []

    for count in range(len(pr_cluster_labels)):

        for j in range(cluster_number_pr):

            forecast_value = 0

            for k in range(cluster_number_z500):

                p_p_given_z = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(k)) & 
                                                       (conditional_probabilities_df["cluster_pr"] == str(j))]['conditional probability'][0]

                p_z = z500_cluster_probabilities[count, k]

                forecast_value = forecast_value + p_p_given_z*p_z

            forecast.append(pd.DataFrame(data={
                    "time": [count],
                    "cluster_pr": [j],
                    "forecast": [forecast_value]}))

    forecast_df = pd.concat(forecast).pivot(index='time', columns='cluster_pr', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count2 in range(len(forecast_df)):
        correct_label_probability = forecast_df.iloc[count2, pr_cluster_labels[count2]]
        bs_add = -1 + 2*correct_label_probability - sos[count2]
        bs = bs + bs_add

    bs = bs/len(forecast_df)

    # now calculate climatological forecast
    
    climatology = []

    for count in range(len(pr_cluster_labels)):

        for j in range(cluster_number_pr):

            forecast_value = pr_cluster_labels[pr_cluster_labels==j].shape[0]/pr_cluster_labels.shape[0]

            climatology.append(pd.DataFrame(data={
                "time": [count],
                "cluster_pr": [j],
                "climatology": [forecast_value]}))

    climatology_df = pd.concat(climatology).pivot(index='time', columns='cluster_pr', values='climatology')

    # calculate sum of squared probabilities of brier skill score
    
    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(pr_cluster_labels)):
        correct_label_probability = climatology_df.iloc[j, pr_cluster_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)

    bss = 1-bs/bsc
    bss
    
    return(bss)




def calculate_cluster_skill_score(cluster_number_pr, cluster_number_z500, pr_cluster_labels, z500_cluster_labels):

    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    conditional_probabilities = []
    for k1 in range(cluster_number_pr):
        for k2 in range(cluster_number_z500):

            P = (pr_cluster_labels[z500_cluster_labels==k2] == k1).mean()

            conditional_probabilities.append(pd.DataFrame(data={
                            "cluster_pr": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [P]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)

    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count in range(len(pr_cluster_labels)):

        for j in range(cluster_number_pr):
            
            forecast_value = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(z500_cluster_labels[count])) & 
                                                       (conditional_probabilities_df["cluster_pr"] == str(j))]['conditional probability'][0]

            forecast.append(pd.DataFrame(data={
                            "time": [count],
                            "cluster_pr": [j],
                            "forecast": [forecast_value]}))
            
    forecast_df = pd.concat(forecast).pivot(index='time', columns='cluster_pr', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    bs = 0
    
    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count2 in range(len(forecast_df)):
        correct_label_probability = forecast_df.iloc[count2, pr_cluster_labels[count2]]
        bs_add = -1 + 2*correct_label_probability - sos[count2]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    
    # now calculate climatological forecast
    
    climatology = []
    
    for count in range(len(pr_cluster_labels)):

        for j in range(cluster_number_pr):
        
            forecast_value = pr_cluster_labels[pr_cluster_labels==j].shape[0]/pr_cluster_labels.shape[0]

            climatology.append(pd.DataFrame(data={
                "time": [count],
                "cluster_pr": [j],
                "climatology": [forecast_value]}))
        
    climatology_df = pd.concat(climatology).pivot(index='time', columns='cluster_pr', values='climatology')
    
    # calculate sum of squared probabilities of brier skill score
    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0
    
    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(pr_cluster_labels)):
        correct_label_probability = climatology_df.iloc[j, pr_cluster_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    bss = 1-bs/bsc
    
    return(bss)





def calculate_95pc_skill_score(cluster_number_z500, z500_cluster_labels, pr_spatial):
    
    pr_95_labels = xr.where(pr_spatial>pr_spatial.quantile(0.95, dim='time'), 1, 0).values
    
    label = z500_cluster_labels
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    conditional_probabilities = []
    for k1 in range(2):
        for k2 in range(cluster_number_z500):

            P = (pr_95_labels[label==k2] == k1).mean()

            conditional_probabilities.append(pd.DataFrame(data={
                            "95pc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [P]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)
    
    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(label)):

        for j in range(2):

            forecast_value = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(label[count2])) & 
                                                       (conditional_probabilities_df["95pc"] == str(j))]['conditional probability'][0]

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "95pc": [j],
                            "forecast": [forecast_value]}))
            
    forecast_df = pd.concat(forecast).pivot(index='time', columns='95pc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_95_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.95]*len(pr_95_labels)
    climatology_df["1"]=[0.05]*len(pr_95_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_95_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)

def calculate_95pc_skill_score_labels(cluster_number_z500, z500_cluster_labels, pr_95_labels):
    
    label = z500_cluster_labels
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    conditional_probabilities = []
    for k1 in range(2):
        for k2 in range(cluster_number_z500):

            P = (pr_95_labels[label==k2] == k1).mean()

            conditional_probabilities.append(pd.DataFrame(data={
                            "95pc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [P]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)
    
    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(label)):

        for j in range(2):

            forecast_value = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(label[count2])) & 
                                                       (conditional_probabilities_df["95pc"] == str(j))]['conditional probability'][0]

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "95pc": [j],
                            "forecast": [forecast_value]}))
            
    forecast_df = pd.concat(forecast).pivot(index='time', columns='95pc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_95_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.95]*len(pr_95_labels)
    climatology_df["1"]=[0.05]*len(pr_95_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_95_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)



def calculate_95pc_skill_score_probabilistic(cluster_number_z500, z500_cluster_probabilities, pr_spatial):
    
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    
    pr_95_labels = xr.where(pr_spatial>pr_spatial.quantile(0.95, dim='time'), 1, 0).values

    conditional_probabilities = []

    for k1 in range(2):
        for k2 in range(cluster_number_z500):

            p3 = z500_cluster_probabilities[:, k2].mean()
            p2 = (pr_95_labels==k1).mean()
            p1 = z500_cluster_probabilities[:, k2][pr_95_labels==k1].mean()

            p = p1*p2/p3

            conditional_probabilities.append(pd.DataFrame(data={
                            "95pc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [p]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)

    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(pr_95_labels)):

        for j in range(2):

            forecast_value = 0

            for k in range(cluster_number_z500):

                p_p_given_z = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(k)) & 
                                                           (conditional_probabilities_df["95pc"] == str(j))]['conditional probability'][0]

                p_z = z500_cluster_probabilities[count2, k]

                forecast_value = forecast_value + p_p_given_z*p_z

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "95pc": [j],
                            "forecast": [forecast_value]}))

    forecast_df = pd.concat(forecast).pivot(index='time', columns='95pc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_95_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.95]*len(pr_95_labels)
    climatology_df["1"]=[0.05]*len(pr_95_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_95_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)


def calculate_95pc_skill_score_probabilistic_labels(cluster_number_z500, z500_cluster_probabilities, pr_95_labels):
    
    # calculate conditional probability of precipitation clusters given a certain z500 cluster

    conditional_probabilities = []

    for k1 in range(2):
        for k2 in range(cluster_number_z500):

            p3 = z500_cluster_probabilities[:, k2].mean()
            p2 = (pr_95_labels==k1).mean()
            p1 = z500_cluster_probabilities[:, k2][pr_95_labels==k1].mean()

            p = p1*p2/p3

            conditional_probabilities.append(pd.DataFrame(data={
                            "95pc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [p]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)

    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(pr_95_labels)):

        for j in range(2):

            forecast_value = 0

            for k in range(cluster_number_z500):

                p_p_given_z = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(k)) & 
                                                           (conditional_probabilities_df["95pc"] == str(j))]['conditional probability'][0]

                p_z = z500_cluster_probabilities[count2, k]

                forecast_value = forecast_value + p_p_given_z*p_z

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "95pc": [j],
                            "forecast": [forecast_value]}))

    forecast_df = pd.concat(forecast).pivot(index='time', columns='95pc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_95_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.95]*len(pr_95_labels)
    climatology_df["1"]=[0.05]*len(pr_95_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_95_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)



def calculate_tercile_skill_score(cluster_number_z500, z500_cluster_labels, pr_spatial):
    
    pr_terc = pd.cut(x = pr_spatial, 
                bins=[-0.0001, np.quantile(pr_spatial, 0.34), np.quantile(pr_spatial, 0.67), np.quantile(pr_spatial, 1)], 
                right=True, labels=list(range(3)), retbins=True, precision=3, include_lowest=True)
    pr_terc_labels = np.array(pr_terc[0]).astype('int')
    
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    
    conditional_probabilities = []
    for k1 in range(3):
        for k2 in range(cluster_number_z500):

            P = (pr_terc_labels[z500_cluster_labels==k2] == k1).mean()

            conditional_probabilities.append(pd.DataFrame(data={
                            "terc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [P]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)
    
    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(pr_terc_labels)):

        for j in range(3):

            forecast_value = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(z500_cluster_labels[count2])) & 
                                                       (conditional_probabilities_df["terc"] == str(j))]['conditional probability'][0]

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "terc": [j],
                            "forecast": [forecast_value]}))
            
    forecast_df = pd.concat(forecast).pivot(index='time', columns='terc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_terc_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.34]*len(pr_terc_labels)
    climatology_df["1"]=[0.33]*len(pr_terc_labels)
    climatology_df["2"]=[0.33]*len(pr_terc_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_terc_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)



def calculate_tercile_skill_score_probabilistic(cluster_number_z500, z500_cluster_probabilities, pr_spatial):
    
    pr_terc = pd.cut(x = pr_spatial, 
                bins=[-0.0001, np.quantile(pr_spatial, 0.34), np.quantile(pr_spatial, 0.67), np.quantile(pr_spatial, 1)], 
                right=True, labels=list(range(3)), retbins=True, precision=3, include_lowest=True)
    
    pr_terc_labels = np.array(pr_terc[0]).astype('int')
    # calculate conditional probability of precipitation clusters given a certain z500 cluster
    
    conditional_probabilities = []
    
    for k1 in range(3):
        for k2 in range(cluster_number_z500):

            p3 = z500_cluster_probabilities[:, k2].mean()
            p2 = (pr_terc_labels==k1).mean()
            p1 = z500_cluster_probabilities[:, k2][pr_terc_labels==k1].mean()

            p = p1*p2/p3

            conditional_probabilities.append(pd.DataFrame(data={
                            "terc": [str(k1)],
                            "cluster_z500": [str(k2)],
                            "conditional probability": [p]}))

    conditional_probabilities_df = pd.concat(conditional_probabilities)
    
    # issue probabilities of pr clusters conditional on observed z500 cluster as forecast
    forecast = []

    for count2 in range(len(pr_terc_labels)):

        for j in range(3):
            
            forecast_value = 0

            for k in range(cluster_number_z500):

                p_p_given_z = conditional_probabilities_df[(conditional_probabilities_df["cluster_z500"] == str(k)) & 
                                                           (conditional_probabilities_df["terc"] == str(j))]['conditional probability'][0]

                p_z = z500_cluster_probabilities[count2, k]

                forecast_value = forecast_value + p_p_given_z*p_z

            forecast.append(pd.DataFrame(data={
                            "time": [count2],
                            "terc": [j],
                            "forecast": [forecast_value]}))
            
    forecast_df = pd.concat(forecast).pivot(index='time', columns='terc', values='forecast')
    
    # calculate sum of squared probabilities of brier skill score
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count3 in range(len(forecast_df)):

        correct_label_probability = forecast_df.iloc[count3, pr_terc_labels[count3]]
        bs_add = -1 + 2*correct_label_probability - sos[count3]
        bs = bs + bs_add

    bs = bs/len(forecast_df)
    
    # now calculate climatological brier score
    
    climatology_df = pd.DataFrame()

    climatology_df["0"]=[0.34]*len(pr_terc_labels)
    climatology_df["1"]=[0.33]*len(pr_terc_labels)
    climatology_df["2"]=[0.33]*len(pr_terc_labels)

    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(climatology_df)):

        correct_label_probability = climatology_df.iloc[j, pr_terc_labels[j]]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = bsc/len(climatology_df)
    
    # calculate full skill score
    
    bss = 1-bs/bsc
    
    return(bss)