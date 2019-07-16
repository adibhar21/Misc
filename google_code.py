# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:57:14 2018

@author: Aditya
"""

def solution(S):
    #HH:MM - 
    #H=H and M=M
    #<60 for MM
    #<24 for HH

    L1 = S.split(':')
    L1 = list(map(int,L1))
    if L1[0]>=24 or L1[1]>=60:
        return "invalid time"
    
    if L1[0]%11==0 and L1[1]%11==0:
        return (":".join(map(str,L1)))
    
    for i in range(L1[0]+1):
        if (L1[0]-i)%11==0:
            L1[0]=L1[0]-i
            break
    for j in range(L1[1]+1):
        if (L1[1]-j)%11==0:
            L1[1]=L1[1]-j
            break
    if(L1[0]<10):
        return(":".join(map(str,L1)))
    return(":".join(map(str,L1)))
            
            

    
    
    
    
    
    
    
    
    
    
    
solution("11:13")