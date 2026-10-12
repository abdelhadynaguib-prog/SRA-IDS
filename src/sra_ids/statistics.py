from __future__ import annotations
from math import comb

def exact_two_sided_mcnemar(b:int,c:int)->float:
    n=int(b+c)
    if n==0:
        return 1.0
    k=min(int(b),int(c))
    lower=sum(comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2.0*lower)

def holm_adjust(p_values):
    p_values=list(map(float,p_values))
    m=len(p_values)
    order=sorted(range(m),key=lambda i:p_values[i])
    out=[None]*m
    running=0.0
    for rank,idx in enumerate(order):
        value=min(1.0,(m-rank)*p_values[idx])
        running=max(running,value)
        out[idx]=running
    return out
