from __future__ import annotations
import numpy as np
import tensorflow as tf
from .constants import LOG_FEATURES, DIRECT_FEATURES

class FeatureIdentityEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_features, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.d_model = int(d_model)
    def build(self, input_shape):
        self.feature_embedding = self.add_weight(
            name="feature_identity_embedding",
            shape=(1, self.n_features, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)
    def call(self, inputs):
        return inputs + self.feature_embedding
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features":self.n_features,"d_model":self.d_model})
        return cfg

def build_verifier(input_dim=20, learning_rate=1e-3):
    d_model, num_heads, key_dim, ff_dim, dropout = 32, 4, 8, 64, 0.10
    inputs = tf.keras.Input(shape=(input_dim,), name="verifier_input")
    x = tf.keras.layers.Reshape((input_dim,1))(inputs)
    x = tf.keras.layers.Dense(d_model)(x)
    x = FeatureIdentityEmbedding(input_dim,d_model)(x)
    for _ in range(2):
        residual = x
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        h = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=key_dim,dropout=dropout
        )(h,h)
        h = tf.keras.layers.Dropout(dropout)(h)
        x = tf.keras.layers.Add()([residual,h])
        residual = x
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        h = tf.keras.layers.Dense(ff_dim,activation="swish")(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        h = tf.keras.layers.Dense(d_model)(h)
        x = tf.keras.layers.Add()([residual,h])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32,activation="swish")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model(inputs,outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model

def signed_log1p(values):
    values = np.asarray(values,dtype=np.float64)
    return np.sign(values)*np.log1p(np.abs(values))

def verifier_numeric_transform(frame):
    out = np.empty((len(frame),len(LOG_FEATURES)+len(DIRECT_FEATURES)),dtype=np.float64)
    for j,feature in enumerate(LOG_FEATURES):
        out[:,j] = signed_log1p(frame[feature].to_numpy(dtype=np.float64))
    offset = len(LOG_FEATURES)
    for j,feature in enumerate(DIRECT_FEATURES):
        out[:,offset+j] = frame[feature].to_numpy(dtype=np.float64)
    return out

def verifier_transform(frame, scaler):
    numeric = verifier_numeric_transform(frame)
    numeric_scaled = scaler.transform(numeric).astype(np.float32)
    proto = frame["proto"].to_numpy(dtype=np.int32)
    if np.any((proto<0)|(proto>4)):
        raise ValueError("Unexpected protocol code.")
    onehot = np.zeros((len(frame),5),dtype=np.float32)
    onehot[np.arange(len(frame)),proto] = 1.0
    transformed = np.concatenate([onehot,numeric_scaled],axis=1).astype(np.float32)
    if transformed.shape[1] != 20:
        raise ValueError(f"Unexpected verifier dimension: {transformed.shape[1]}")
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite verifier transformed values.")
    return transformed
