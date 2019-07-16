# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 18:25:16 2018

@author: Aditya
"""
# 1) Write three functions that compute the sum of the numbers in a given list using a for-loop, a while-loop, and recursion. 
def forloop(lst):
    current = result = 0
    for i in lst:
        current = i
        result = result+current
    return result

def whileloop(lst):
    i = result = 0
    while i<len(lst):        
        result = result + lst[i]
        i = i + 1
    return result


def recursion(lst):
    if len(lst)<1:
        return 0
    return lst[0]+recursion(lst[len(lst)-(len(lst)-1):])


# 2) Write a function that combines two lists by alternatingly taking elements. For example: given the two lists [a, b, c] and [1, 2, 3], the function should return [a, 1, b, 2, c, 3]. 


def twolists(lst,lst1):
    if len(lst)<1 and len(lst1)<1:
        return []
    lst2 = [lst]+[lst1]
    return list(filter(lambda i:i ,[i[0] for i in lst2])) + twolists(lst[len(lst)-(len(lst)-1):],lst1[len(lst)-(len(lst)-1):])

# 3) Write a function that computes the list of the first 100 Fibonacci numbers. By definition, the first two numbers in the Fibonacci sequence are 0 and 1, and each subsequent number is the sum of the previous two. As an example, here are the first 10 Fibonnaci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34

d1 = {}
def fib(n):
    if n==1:
        return 1
    if n==0:
        return 0
    if not n in d1:
        d1[n] = fib(n-1) + fib(n-2)
    return d1[n]