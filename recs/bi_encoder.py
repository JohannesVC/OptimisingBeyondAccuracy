import keras
from keras.api import layers
from recs.subclasses import *
from recs.data_loader import DataLoader

class OptimisedModel(keras.Model):
    """
    The OptimisedModel is a bi-encoder recommendation model leveraging attention mechanisms for encoding user history 
    and news content, with additional regularization for diversity and novelty.

    This model is designed to optimize beyond accuracy by incorporating diversity and novelty 
    regularization into the recommendation process. It uses multi-head attention mechanisms 
    to encode news content and user history, and supports both BERT and GloVe embeddings.

    Usage:
        - Instantiate the BiRec model with the desired parameters.
        - Train the model using a suitable dataset and optimizer.
        - Use the trained model to generate recommendations for users.
    Note:
        - The model supports both BERT and GloVe embeddings for encoding news content.
        - Regularization layers for diversity and novelty are included to enhance 
          recommendation quality beyond accuracy.
    """
    def __init__(self,                
                 data: DataLoader,
                 dropout=0.2,
                 head_num=16,
                 head_dim=64,
                 intermediate_dim=200,
                 lambda_diversity=1e2,
                 lambda_novelty=0.1,              
                 dataset_size='small',
                 embeddings='bert',
                 train_embeds=False,
                 batch_size=128,
                 max_impressions=25,
                 max_history_length=25,
                 max_title_length=20,
                 name="nmrs_rank", 
                 **kwargs):
        """
        Initialize the bi-encoder recommendation model. 

        Args:
            dropout (float): Dropout rate for regularization.
            head_num (int): Number of attention heads in the multi-head attention mechanism.
            head_dim (int): Dimension of each attention head.
            intermediate_dim (int): Dimension of the intermediate dense layer in the user encoder.
            lambda_diversity (float): Regularization weight for diversity.
            lambda_novelty (float): Regularization weight for novelty.
            dataset_size (str): Descriptor of the dataset size ('small', 'large').
            embeddings (str): Type of embeddings to use ('bert' or 'glove').
            train_embeds (bool): Whether to fine-tune the embedding layer during training.
            batch_size (int): Number of samples per batch during training.
            max_impressions (int): Maximum number of impressions per sample.
            max_history_length (int): Maximum length of the user history sequence.
            max_title_length (int): Maximum length of the news title sequence.

        Methods:
            __init__(data, dropout, head_num, head_dim, intermediate_dim, lambda_diversity, 
                    lambda_novelty, dataset_size, embeddings, train_embeds, batch_size, 
                    max_impressions, max_history_length, max_title_length, name, **kwargs):
                Initializes the BiRec model with the specified parameters.
            _build_news_encoder():
                Builds the news encoder sub-model, which encodes news titles using embeddings 
                and attention mechanisms.
            _build_user_encoder():
                Builds the user encoder sub-model, which encodes user history by applying 
                attention mechanisms over the encoded news titles.
            call(inputs, training=False):
                Performs a forward pass through the model. Encodes user history and candidate 
                news items, applies diversity regularization, and computes the final 
                recommendation scores.
        """
        super().__init__(name=name, **kwargs)
        
        self.dropout = dropout
        self.head_num = head_num
        self.head_dim = head_dim   
        self.intermediate_dim = intermediate_dim
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.max_impressions = max_impressions
        self.max_history_length = max_history_length   
        self.max_title_length = max_title_length
        self.embeddings = embeddings

        # Instantiate embedding layers and optional `NoveltyRegularisation` layer
        if embeddings == 'bert':
            self.embedding_layer = BERTEmbedding(dataset_size, train_embeds)
            data.get_popularity()
            self.novelty_layer = NoveltyRegulariser(
                data.item_popularity, data.total_users, lambda_novelty)    
        else:
            self.embedding_layer = GloveEmbedding(dataset_size, train_embeds)
            self.novelty_layer = layers.Lambda(lambda x: x)

        # Build sub-models
        self._news_encoder = self._build_news_encoder()
        self._user_encoder = self._build_user_encoder()

        # Build final model layers
        self.time_dist = layers.TimeDistributed(self._news_encoder)
        self.dot = layers.Dot(-1)
        self.activation = layers.Softmax()

        # Regularisation
        self.diversity_layer = DiversityRegularisation(lambda_diversity=lambda_diversity)

    def _build_news_encoder(self):
        title_input = keras.Input(
            shape=(self.max_title_length,), dtype="int32", name='news_encoder_input')
        embedded_title = self.embedding_layer(title_input) 
        y = layers.Dropout(self.dropout)(embedded_title) 
        y = layers.MultiHeadAttention(self.head_num, self.head_dim, 
                                      name='news_attention')(y, y)
        y = layers.Dropout(self.dropout)(y)        
        title_vector = AttentionPooling()(y) 
        return keras.Model(title_input, title_vector, name="news_encoder")
    
    def _build_user_encoder(self):
        history_input = keras.Input(
            shape=(self.max_history_length, self.max_title_length), dtype="int32")
        y = layers.TimeDistributed(self._news_encoder)(history_input)
        y = layers.MultiHeadAttention(self.head_num, self.head_dim, 
                                      dropout=self.dropout, 
                                      name='user_attention')(y, y)
        
        z = layers.Dense(self.intermediate_dim, activation='relu')(y)
        z = layers.Dense(ops.shape(y)[-1])(z)
        z = layers.Add()([y, z])
        encoded_user = AttentionPooling()(z)
        return keras.Model(history_input, encoded_user, name="user_encoder")
    
    def call(self, inputs, training=False):
        history_input, candidate_input = inputs 

        if self.embeddings == 'bert': 
            history_input = ops.expand_dims(history_input, -1)
            candidate_input = ops.expand_dims(candidate_input, -1)
        
        encoded_user = self._user_encoder(history_input)        # (batch, dims)
        encoded_candidates = self.time_dist(candidate_input)    # (batch, num_candidates, dims) !!
        
        logits = self.dot([encoded_candidates, encoded_user])   

        logits = self.diversity_layer(logits, encoded_candidates) 
        logits = self.novelty_layer(logits, candidate_input) 

        outputs = self.activation(logits)
        return outputs
    