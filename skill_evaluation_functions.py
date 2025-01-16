#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:46:38 2024

@author: fionaspuler
"""

import xarray as xr
import pandas as pd
import numpy as np

def precip_cluster_forecast_proba(conditional_probabilities_df, cluster_number_pr, cluster_number_z500, 
                                                z500_cluster_probabilities):
    
    # issue forecast based on conditional probabilities
    forecast = []

    for count in range(len(z500_cluster_probabilities)):

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
    return(forecast_df)


def calculate_conditional_probabilities_proba(cluster_number_pr, cluster_number_z500, 
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
    return(conditional_probabilities_df)

def calculate_conditional_probabilities(cluster_number_pr, cluster_number_z500, 
                                        pr_cluster_labels, z500_cluster_labels):

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
    
    return(conditional_probabilities_df)


# functions to calculate brier scores

def calculate_brier_skill_score_clusters_probabilistic(y_true, y_pred_proba):
    
    forecast_df = pd.DataFrame(y_pred_proba)
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count2 in range(len(forecast_df)):
        correct_label_probability = forecast_df.iloc[count2, y_true[count2].astype('int')]
        bs_add = -1 + 2*correct_label_probability - sos[count2]
        bs = bs + bs_add

    bs = -bs/len(forecast_df)

    return(bs)


def calculate_brier_skill_score_clusters(y_true, y_pred_df, label_name):
    
    y_pred_df['value']=1
    forecast_df = y_pred_df.pivot(columns=label_name, values='value')
    forecast_df = forecast_df.fillna(0)
    
    squared=forecast_df.apply(lambda num: num**2)
    sos=squared.sum(axis=1)
    bs = 0

    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for count2 in range(len(forecast_df)):
        
        correct_label_probability = forecast_df.iloc[count2, y_true[count2].astype('int')]
        bs_add = -1 + 2*correct_label_probability - sos.values[count2]
        bs = bs + bs_add

    bs = -bs/len(forecast_df)
    return(bs)


def brier_score_cluster_climatology(era5_labels, k):
    
    climatology = []
    
    for count in range(len(era5_labels)):

        for j in range(k):
        
            forecast_value = era5_labels[era5_labels==j].shape[0]/era5_labels.shape[0]

            climatology.append(pd.DataFrame(data={
                "time": [count],
                "cluster_era5": [j],
                "climatology": [forecast_value]}))
        
    climatology_df = pd.concat(climatology).pivot(index='time', columns='cluster_era5', values='climatology')
    
    # calculate sum of squared probabilities of brier skill score
    squared2=climatology_df.apply(lambda num: num**2)
    sos2=squared2.sum(axis=1)

    bsc = 0
    
    # for each time step calculate brier skill score, add to total and divide by number of timesteps in the end
    for j in range(len(era5_labels)):
        correct_label_probability = climatology_df.iloc[j, era5_labels[j].astype('int')]
        bs_add = -1 + 2*correct_label_probability - sos2[j]
        bsc = bsc + bs_add

    bsc = -bsc/len(climatology_df)
    return(bsc)



