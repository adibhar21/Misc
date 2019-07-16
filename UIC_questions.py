# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:42:50 2018

@author: Aditya
"""
import numpy as ny
import pandas as pd

#question 5 section 1

def open_file():
    file = "C:/Users/Aditya/Desktop/pranav/pranav.csv" #use your file address!
    file_line = open(file, 'r')
    return [i.replace('\n','').split(',') for i in file_line]

# section 2
    
#question 1
ndarray = ny.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
def access_numpy(ndarray):
    ndarray[:] = [2,2,2,2,2]
    for i in range(ny.int(ndarray.size/5)):
        ndarray[i][::2] = 3
    return ndarray

#question 2
def read_fromtxt_using_numpy():
    file = "C:/Users/Aditya/Desktop/pranav/MidTerm1/nyc_taxis.csv"
    nyc_taxis = ny.genfromtxt(file,dtype=str,delimiter=",") #dtype needs to be checked!
    return nyc_taxis

#question 3
file = "C:/Users/Aditya/Desktop/pranav/MidTerm1/nyc_taxis.csv"
nyc_taxis = ny.genfromtxt(file,dtype=str,delimiter=",") #dtype needs to be checked!
def concatenate_two_arrays(nyc_taxis):
    modified_taxi = ny.concatenate((nyc_taxis,ny.zeros((89561,1),dtype=int)),axis=1)
    return modified_taxi

modified_taxi = ny.concatenate((nyc_taxis,ny.zeros((89561,1),dtype=int)),axis=1)
def boolean_indexing(modified_taxi):
    for value in range(1,89561):
        if ny.equal(ny.int(modified_taxi[value,5]),2):
            modified_taxi[value,15]=1
        elif ny.equal(ny.int(modified_taxi[value,5]),3):
            modified_taxi[value,15]=1
        elif ny.equal(ny.int(modified_taxi[value,5]),5):
            modified_taxi[value,15]=1
    return modified_taxi[:,15]


#............Section 1..................

#..........question 1...........
    
def list_comprehensions(stock_prices):
    updated_stock_price = [price+1 if price%2==0 else price for price in stock_prices]
    return updated_stock_price

def group_items(vehicles):
    {value:key for key:value in }


def read_and_return_col_names():
    stats_f500 = pd.read_csv('C:\\Users\\Aditya\\Desktop\\pranav\\MidTerm1\\f500.csv')
    stats_f500.iloc[:,6] = ny.NAN
    stats_f500 = stats_f500.fillna(method='ffill')







#question 5

def open_file():
    file = "C:/Users/Aditya/Desktop/pranav/pranav.csv" #use your file address!
    file_line = open(file, 'r')
    return [i.replace('\n','').split(',') for i in file_line]