from __future__ import annotations
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import tensorflow as tf
from .constants import FEATURES,PRIMARY_THRESHOLD,ROUTER_PROB_THRESHOLD,ROUTER_NN1_THRESHOLD,VERIFIER_THRESHOLD
from .preprocessing import clean_bot_iot_frame,make_routing_representation
from .verifier import FeatureIdentityEmbedding,verifier_transform

class FrozenSRAIDS:
    def __init__(self,artifact_dir):
        artifact_dir=Path(artifact_dir)
        self.primary=lgb.Booster(model_file=str(artifact_dir/"final_primary_lightgbm.txt"))
        self.router_scaler=joblib.load(artifact_dir/"final_router_standard_scaler.joblib")
        self.normal_nn=joblib.load(artifact_dir/"final_normal_reference_nn.joblib")
        self.normal_leaf_sets=joblib.load(artifact_dir/"final_normal_leaf_sets.joblib")
        self.verifier_scaler=joblib.load(artifact_dir/"final_verifier_robust_scaler.joblib")
        self.verifier=tf.keras.models.load_model(
            artifact_dir/"final_residual_attention.keras",
            custom_objects={"FeatureIdentityEmbedding":FeatureIdentityEmbedding},
            compile=False,
        )

    def _leaf_match_rate(self,x_clean):
        leaves=np.asarray(self.primary.predict(x_clean[FEATURES],pred_leaf=True))
        if leaves.ndim==1:
            leaves=leaves[:,None]
        if leaves.shape[1]!=len(self.normal_leaf_sets):
            raise RuntimeError("Leaf-set count does not match LightGBM tree count.")
        count=np.zeros(len(x_clean),dtype=np.int32)
        for tree_id,normal_set in enumerate(self.normal_leaf_sets):
            count += np.isin(leaves[:,tree_id],normal_set)
        return (count/float(leaves.shape[1])).astype(np.float32)

    def predict_frame(self,raw_frame,batch_size=1024):
        x=clean_bot_iot_frame(raw_frame)
        primary_p=np.asarray(self.primary.predict(x[FEATURES]),dtype=np.float64)
        primary_pred=(primary_p>=PRIMARY_THRESHOLD).astype(np.uint8)

        routing_x=make_routing_representation(x,self.router_scaler)
        distances,_=self.normal_nn.kneighbors(routing_x,n_neighbors=3,return_distance=True)
        nn1=distances[:,0].astype(np.float32)
        nn3=distances.mean(axis=1).astype(np.float32)
        leaf_match=self._leaf_match_rate(x)

        routed=(primary_p<ROUTER_PROB_THRESHOLD)|(nn1<=ROUTER_NN1_THRESHOLD)
        verifier_p=np.full(len(x),np.nan,dtype=np.float64)
        final_pred=primary_pred.copy()

        idx=np.flatnonzero(routed)
        if idx.size:
            vf=x.iloc[idx].copy()
            vf["primary_p_attack"]=primary_p[idx]
            vf["leaf_match_rate"]=leaf_match[idx]
            vf["NN1_distance"]=nn1[idx]
            vf["NN3_distance"]=nn3[idx]
            xv=verifier_transform(vf,self.verifier_scaler)
            p=self.verifier.predict(xv,batch_size=batch_size,verbose=0).reshape(-1).astype(np.float64)
            verifier_p[idx]=p
            final_pred[idx]=(p>=VERIFIER_THRESHOLD).astype(np.uint8)

        return {
            "prediction":final_pred,
            "primary_p_attack":primary_p,
            "primary_prediction":primary_pred,
            "routed":routed,
            "verifier_p_attack":verifier_p,
            "leaf_match_rate":leaf_match,
            "NN1_distance":nn1,
            "NN3_distance":nn3,
        }
