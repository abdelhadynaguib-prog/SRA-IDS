from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, f1_score
)

def binary_metrics(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=np.uint8)
    y_pred=np.asarray(y_pred,dtype=np.uint8)
    tn=int(((y_true==0)&(y_pred==0)).sum())
    fp=int(((y_true==0)&(y_pred==1)).sum())
    fn=int(((y_true==1)&(y_pred==0)).sum())
    tp=int(((y_true==1)&(y_pred==1)).sum())
    return {
        "TN":tn,"FP":fp,"FN":fn,"TP":tp,"Errors":fp+fn,
        "Accuracy":float(accuracy_score(y_true,y_pred)),
        "Balanced_Accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "MCC":float(matthews_corrcoef(y_true,y_pred)),
        "Normal_Precision":float(precision_score(y_true,y_pred,pos_label=0,zero_division=0)),
        "Normal_Recall":float(recall_score(y_true,y_pred,pos_label=0,zero_division=0)),
        "Normal_F1":float(f1_score(y_true,y_pred,pos_label=0,zero_division=0)),
        "Attack_Precision":float(precision_score(y_true,y_pred,pos_label=1,zero_division=0)),
        "Attack_Recall":float(recall_score(y_true,y_pred,pos_label=1,zero_division=0)),
        "Attack_F1":float(f1_score(y_true,y_pred,pos_label=1,zero_division=0)),
        "FPR":float(fp/(fp+tn)) if fp+tn else float("nan"),
        "FNR":float(fn/(fn+tp)) if fn+tp else float("nan"),
    }

def wilson_ci(successes,total,z=1.959963984540054):
    if total<=0:
        return float("nan"),float("nan")
    p=successes/total
    denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return float(center-half),float(center+half)
