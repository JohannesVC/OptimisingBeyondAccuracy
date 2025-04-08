"""
Custom Subclasses Module
------------------------
This module provides custom Keras components for ranking, recommendation, and embedding tasks.

It includes:
    - TestRegularizer: A simple test layer that adds a constant regularization loss.
    - DiversityRegularisation: Penalizes high cosine similarity among embeddings to encourage diversity.
    - NoveltyRegulariser: Boosts novelty by rewarding surprisal/self-information based on item popularity.
    - PopularityPenalty: Penalizes overuse of popular items in top-k recommendations.
    - MaskedGlobalAveragePooling1D: Computes a masked global average pooling operation, ignoring padded values.
    - AttentionPooling: Applies an attention mechanism to pool a sequence of feature vectors into a single representation.
    - ClickLoss: A custom loss function for click prediction using binary cross-entropy, ignoring padded values.
    - RankLoss: A custom loss function for ranking tasks using categorical cross-entropy, ignoring padded values.
    - MaskedAUC: Computes the Area Under the ROC Curve (AUC) while ignoring padded values in the ground truth labels.
    - NDCGScore: Computes the Normalized Discounted Cumulative Gain (NDCG) for ranking tasks, comparing predicted and ideal rankings.
    - GloveEmbeddingLookup: A custom layer for embedding lookup with pre-trained GloVe embeddings.
    - GloveEmbedding: A layer for loading and using pre-trained GloVe embeddings with masking support.
    - EmbeddingLookup: A custom layer for embedding lookup with pre-trained embeddings and automatic masking.
"""

import keras
import tensorflow as tf
from keras.api import ops
import numpy as np
from pathlib import Path

class TestRegularizer(keras.layers.Layer):
    """Layer that creates an activity sparsity regularization loss."""
    def __init__(self):
        super().__init__() 

    def call(self, inputs):
        self.add_loss(tf.constant(1, dtype=tf.float32))
        return inputs  

class DiversityRegularisation(keras.layers.Layer):
    """Penalises high cosine similarity among embeddings."""    
    def __init__(self, lambda_diversity=0.01, warmup_epoch=1, **kwargs):
        super().__init__(name='DiversityRegularisation', **kwargs)
        self.lambda_diversity = lambda_diversity         
        
    def call(self, logits, encoded_candidates): 

        # Normalize all candidate vectors
        news_norm = tf.nn.l2_normalize(encoded_candidates, axis=-1)  # (B, Cand, D)

        # Compute full similarity matrix
        sim_matrix = tf.matmul(news_norm, news_norm, transpose_b=True)  # (B, Cand, Cand)

        # Create pairwise weight matrix
        temperature = 0.1 # to approximate hard top‑k focus
        probs = ops.softmax(logits / temperature, axis=-1)    
        p_i = ops.expand_dims(probs, axis=2)               
        p_j = ops.expand_dims(probs, axis=1)                   
        pairwise_weights = p_i * p_j                           

        # Mask out the diagonal (self-similarity)
        mask = tf.eye(ops.shape(sim_matrix)[1], batch_shape=[ops.shape(sim_matrix)[0]])
        masked_sim = sim_matrix * (1.0 - mask)

        # Weighted expected similarity
        expected_similarity = ops.sum(pairwise_weights * masked_sim, axis=[1, 2])
        mean_div_loss = self.lambda_diversity * ops.mean(expected_similarity)
        
        self.add_loss(mean_div_loss)

        if tf.executing_eagerly():
            print(f"Diversity: {mean_div_loss}") 

        return logits  
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'div_weight': self.lambda_diversity,
        })
        return config


class NoveltyRegulariser(keras.layers.Layer):
    """Boosts Novelty aka surprisal."""    
    def __init__(self, item_popularity, total_users, lambda_novelty=1.0, topk=5, **kwargs):
        super().__init__(name='NoveltyRegularisation', **kwargs)

        max_item_id = max(item_popularity.keys()) + 1
        pop_list = [item_popularity.get(i, 0) for i in range(max_item_id)]
        self.item_popularity_tensor = tf.constant(pop_list, dtype=tf.float32)
        self.total_users = total_users
        self.lambda_novelty = lambda_novelty
        self.topk = topk 

    def call(self, logits, candidate_news): 
        
        # Use probs to make differentiable
        temperature = 0.1 # to approximate hard top‑k focus
        probs = ops.softmax(logits / temperature, axis=-1) 

        # Take 
        pop_scores = tf.gather(self.item_popularity_tensor, ops.squeeze(candidate_news, -1))

        # Self-information (aka surprisal)
        surprise_scores = ops.log(self.total_users / (pop_scores + 1.0))

        # Weighted average (per user)
        expected_surprise = ops.sum(probs * surprise_scores, axis=-1)

        # Reward surprise (= higher is better)
        novelty_loss = -self.lambda_novelty * ops.mean(expected_surprise)
       
        self.add_loss(novelty_loss)

        if tf.executing_eagerly():
            print(f"Surprisal: {novelty_loss}") 

        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'topk': self.topk,
            'div_weight': self.lambda_novelty,
        })
        return config
    
class PopularityPenalty(keras.layers.Layer):
    """Encourages the model to spread its recommendations across different item"""
    def __init__(self, lambda_popularity=1.0, topk=10, **kwargs):
        super().__init__(**kwargs)
        self.lambda_popularity = tf.constant(lambda_popularity, dtype=tf.float32)
        self.topk = topk     

    def call(self, logits, candidate_news):        

        # Count the number of occurences of the same candidates in the batch
        flat_ids = ops.reshape(candidate_news, (-1,)) 
        counts = ops.bincount(flat_ids)

        # High counts = items appearing too often in the top-k lists
        overuse_penalty = ops.sum(ops.square(tf.cast(counts, tf.float32)))
        overuse_penalty = self.lambda_popularity * overuse_penalty

        self.add_loss(overuse_penalty)

        if tf.executing_eagerly():
            print(f"overuse_penalty: {overuse_penalty}") 

        return logits


class MaskedGlobalAveragePooling1D(keras.Layer):
    """Masks out -1s before averaging"""
    def __init__(self, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims
        self.supports_masking = True

    def call(self, inputs, mask=None):        
        
        if mask is None: # Return a simple mean
            return ops.mean(inputs, axis=1, keepdims=self.keepdims)
        
        # if tf.executing_eagerly:
        #     print(mask.shape)

        # Compute mask
        mask = ops.cast(mask, inputs.dtype)
        mask = ops.expand_dims(mask, axis=-1)
        masked_sum = ops.sum(inputs * mask, axis=1, keepdims=self.keepdims)
        count = ops.sum(mask, axis=1, keepdims=self.keepdims)
        return masked_sum / ops.maximum(count, tf.constant(1, dtype=inputs.dtype))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "keepdims": self.keepdims,
            "supports_masking": self.supports_masking,
        })
        return config
    
class SoftAttention(keras.Layer):
    """Soft attention with scalar scoring"""
    def __init__(self, name="SoftAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True
        self.attention_logits = keras.layers.Dense(1, activation='tanh')         

    def call(self, inputs, mask=None):
        attention_scores = self.attention_logits(inputs)

        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.float32)
            # Set scores to large negative number where mask == 0
            attention_scores += (1.0 - mask) * -1e9

        attention_weights = ops.softmax(attention_scores, axis=1) 
        context_vector = ops.sum(inputs * attention_weights, axis=1) 
        return context_vector

class AttentionPooling(keras.Layer):
    """
    Attention pooling mechanism reduce a sequence of feature vectors 
    to one contextualised vector
    """
    def __init__(self, name="AttentionPooling", **kwargs): # query_dim = 200, 
        super().__init__(name=name, **kwargs)
        # self.query_dim = query_dim 

    def build(self, inputs_shape):        
        feature_dim = inputs_shape[-1]
        self.W = self.add_weight(name='W',
                                 shape=(feature_dim, feature_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(feature_dim,),
                                 initializer='zeros',
                                 trainable=True)
        self.q = self.add_weight(name='q',
                                 shape=(feature_dim,1), 
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionPooling, self).build(inputs_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        # Compute attention scores
        attn = ops.tanh(ops.matmul(inputs, self.W) + self.b)
        attn = ops.matmul(attn, self.q)
        attn = ops.squeeze(attn, axis=-1)   

        # If a mask is provided, use it
        if mask is not None:
            # if tf.executing_eagerly: print(mask.shape)
            if len(mask.shape) == 3:
                mask = ops.squeeze(mask, axis=1)
            attn = ops.exp(attn) * ops.cast(mask, dtype=attn.dtype)
        else:
            attn = ops.exp(attn)

        # Normalise the attention scores
        attn_weights = attn / (ops.sum(attn, axis=-1, keepdims=True) + keras.backend.epsilon())
        attn_weights = ops.expand_dims(attn_weights, axis=-1)

        # Weighted sum
        pooled_output = ops.sum(inputs * attn_weights, axis=1)
        pooled_output = ops.copy(pooled_output) 
        return pooled_output

    def compute_mask(self, inputs, mask=None):
        # No mask is needed downstream, 
        # but keras throws an error if it doesn't contain compute_mask
        return None
    
    def compute_output_shape(self, input_shape):
        # Output: (batch, feature_dim)
        return (input_shape[0], input_shape[-1])    

def minus_one_mask(y_true, y_pred):
    # Flatten
    y_true = ops.reshape(y_true, [-1])  
    y_pred = ops.reshape(y_pred, [-1]) 
    
    # Create a mask to ignore padded values (-1)
    mask = ops.not_equal(y_true, -1)

    # Take the indices corresponding to the mask
    y_pred = ops.take(y_pred, ops.where(mask)[0])
    y_true = ops.take(y_true, ops.where(mask)[0])
    return y_true, y_pred

class ClickLoss(keras.Loss):
    """Custom loss function for handling click prediction."""
    def __init__(self,name="click_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred): 
        loss = keras.losses.binary_crossentropy(*minus_one_mask(y_true, y_pred))       
        return ops.mean(loss)

class RankLoss(keras.Loss):   
    def __init__(self, name="rank_loss", **kwargs):
        super().__init__(name=name, **kwargs) 

    def call(self, y_true, y_pred):
        loss = keras.losses.categorical_crossentropy(*minus_one_mask(y_true, y_pred))
        return ops.mean(loss)

class MaskedAUC(keras.metrics.Metric):
    """
    Computes the Area Under the ROC Curve (AUC) while ignoring padded values in the ground truth labels. 
    Padded values are assumed to be represented by -1, and they are excluded from the AUC computation by applying a boolean mask.
    This metric wraps the standard Keras AUC metric, updating it only with the non-padded
    values of y_true and y_pred.
    """
    def __init__(self, name="masked_auc", **kwargs):       
        super().__init__(name=name, **kwargs)
        self.auc = keras.metrics.AUC(from_logits=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.auc.update_state(*minus_one_mask(y_true, y_pred))

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()    

class GloveEmbedding(keras.layers.Layer):
    def __init__(self, dataset_size='small', trainable=False, **kwargs):
        super().__init__(name="glove_embedding", **kwargs) 
        
        embedding_matrix = np.load('../.data/glove_embedding_matrix_300_small.npy')
        if dataset_size == 'large':
            embedding_matrix =  np.load('../.data/glove_embedding_matrix_300_large.npy')

        input_dim, embed_dim = embedding_matrix.shape
        self.embedding_layer = keras.layers.Embedding(
            input_dim, embed_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=trainable,
            mask_zero=True,
            # embeddings_regularizer=keras.regularizers.L2(1e-3) 
        )
    def call(self, inputs):        
        return self.embedding_layer(inputs)

class BERTEmbedding(keras.layers.Layer):
    """
    A custom Keras layer for BERT word-level embedding lookup with pre-trained embeddings and masking.
    This layer loads pre-trained embeddings and a corresponding mask from a `.npz` file.
    It allows embedding lookup for input indices and automatically computes a mask
    for the inputs based on the stored mask matrix.
    """       
    def __init__(self, dataset_size='small', trainable=False, **kwargs):
        super().__init__(name="BERT_embedding", **kwargs)
        
        if dataset_size == 'small':
            path = '../.data/bert_emb_mask_float32_256d_small_mask.npz' 
        elif dataset_size == 'large':
            path = '../.data/bert_emb_mask_float32_256d_large_mask.npz' 
        data = np.load(path)
        
        self.embedding_matrix = data["embeddings"] # (num_words, embedding_dim)
        self.mask_matrix = data["mask"].astype(bool)        # (num_words,)

        # Store embeddings as a TensorFlow variable
        self.embedding_matrix_var = self.add_weight(
            name="embedding_matrix",
            shape=self.embedding_matrix.shape,
            initializer=tf.constant_initializer(self.embedding_matrix),
            trainable=trainable,
        )

    def call(self, inputs):
        """Perform embedding lookup and automatically extract mask """
        inputs = ops.squeeze(inputs, axis=-1)     
        return ops.take(self.embedding_matrix_var, inputs, axis=0)

    def compute_mask(self, inputs, mask=None):
        """
        Automatically infer the mask from the stored mask matrix
        This allows propagation
        """
        return ops.take(self.mask_matrix, inputs, axis=0)
