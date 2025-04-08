"""
Data Module
-----------------
This module provides a robust and efficient framework for loading and processing the MIND dataset for news recommendation tasks. Designed for integration with Keras, it leverages Keras’s TextVectorizer and the tf.data.Dataset API to streamline tokenization, dynamic sequence padding, and data segmentation.

Key Features:
    - Efficient parsing and indexing of news articles and user behavior logs.
    - Dynamic padding of user history and candidate impression sequences.
    - Seamless integration with pre-trained embeddings using Keras's TextVectorizer.
    - Flexible mapping functions to support both raw text strings and tokenized representations.
    - Easy retrieval of training, validation, and testing datasets via dedicated methods.

Usage:
    1. Instantiate the DataLoader with the preferred parameters (e.g., sequence lengths, data size).
    2. Load and index news data using init_news().
    3. Process user behaviors with init_behaviors() to generate padded sequences.
    4. Obtain the desired dataset by calling get_train(), get_val(), or get_test().

This module replaces earlier an notebook-based implementation by offering a faster and more integrated approach to data handling, strictly tailored for Keras-based pipelines.
"""

from pathlib import Path
from typing import Literal
from keras.api.preprocessing.sequence import pad_sequences
from keras.api import layers, ops
import tensorflow as tf 
import numpy as np

class DataLoader:
    """
    DataLoader class responsible for loading and processing news and behavior datasets for a recommendation system.
    
    Usage Example:
        loader = DataLoader(max_hist_len=50, max_imps=20, data_size='small')
        news_titles = loader.init_news(news_file=loader.news_train)
        impressions, labels, histories, user_indices = loader.init_behaviors(behaviors_file=loader.behaviors_train)
        train_dataset = loader.get_train(vectorizer=my_vectorizer)
    """
    def __init__(self, 
                 batch_size: int = 256, 
                 max_hist_len: int = 25, 
                 max_imps: int = 25,             
                 data_size: Literal['small', 'large'] = 'small'
                 ):
        """
        Initializes the data loader with configuration parameters for processing news and behavior datasets.
        Args:
            max_hist_len (int): Determines the padding length for history sequences. Across the MIND dataset, median = 20, mean = 34, and 3rd quartile is 43.  
            max_imps (int): Determines the padding length for candidate impression sequences and labels. Across the MIND dataset, median = 24, mean = 37, and 3rd quartile is 51.  
            data_size (Literal['small', 'large']): Specifies the dataset type to use ('small' or 'large'). Default is 'small'.
        Attributes:
            news_id2index (dict): Maps news IDs to corresponding indices.
            max_hist_len (int): Stores the maximum history length.
            max_imps (int): Stores the maximum number of impressions.
            data_size (Literal['small', 'large']): Stores the chosen dataset type.
            news_train (Path): Path to the training news data.
            behaviors_train (Path): Path to the training behaviors data.
            news_dev (Path): Path to the development news data.
            behaviors_dev (Path): Path to the development behaviors data.
            news_test (Path): Path to the test news data.
            behaviors_test (Path): Path to the test behaviors data.
        The file paths for the news and behavior data are set based on the value of data_size:
            - For 'small', the paths are assigned from the corresponding small dataset locations.
            - For 'large', the paths are assigned from the corresponding large dataset locations.
            An instance of the data loader with its attributes initialized.
        """
        
        self.news_id2index = {}
        self.max_hist_len = max_hist_len
        self.max_imps = max_imps
        self.data_size = data_size
        self.logging_max_idx = 0 # to keep track of alignment
        self.vectorizer = None
        self.K = None
       
        BASE = Path('/home/jonux/G/DataScienceMSc/final_project/project_folder')

        if self.data_size == 'small':
            
            self.train_news = BASE / ".data/MINDsmall_train/news.tsv"
            self.train = BASE / ".data/MINDsmall_train/behaviors.tsv"
            
            self.dev_news = self.test_news = BASE / ".data/MINDsmall_dev/news.tsv"
            self.dev = self.test = BASE / ".data/MINDsmall_dev/behaviors.tsv"  
        else:    
            self.train = BASE / ".data/MINDlarge_train/behaviors.tsv"
            self.train_news = BASE / ".data/MINDlarge_train/news.tsv"

            self.dev_news = self.test_news = BASE / ".data/MINDlarge_dev/news.tsv"
            self.dev = BASE / ".data/MINDlarge_new_dev/behaviors.tsv" # split in dataloader.ipynb
            self.test = BASE / ".data/MINDlarge_new_test/behaviors.tsv"

        self.init_news(self.train_news) 
        self.update_news(self.dev_news)

        self.batch_size = batch_size


    def init_news(self, news_file):  
        """
        Initialise the news titles and update the news_id to index mapping from a given file.
        This method processes a file containing news records, where each record is expected to have at least four
        tab-separated values. It extracts the news ID (from the first column) and the news title (from the fourth column)
        for each valid line, and updates the mapping from news IDs to their corresponding indices while also building a
        list of news titles. Index 0 of the list is intentionally set to an empty string to act as padding.
        Parameters:
            news_file (str): The file path to the news data file. Each line in the file should contain tab-separated fields.
        Returns:
            List[str]: A list of news titles with the first element as an empty string for padding purposes.
        Notes:
            - Lines with fewer than four columns are skipped.
            - The index assigned to each news ID starts at 1 to preserve the 0 index for the padding.
        """
        
        self.news_titles = [''] # ensures padding corresponds to index 0
        with open(news_file, "r") as f:
            for line in f:
                columns = line.strip().split("\t")
                if len(columns) < 4:
                    continue
                title = columns[3]
                news_id = columns[0]
                self.news_id2index[news_id] = len(self.news_id2index) + 1   # +1 to preserve 0 for padding
                self.news_titles.append(title)
        return self.news_titles
    
    def update_news(self, news_file):
        """
        Similar to init_news() 
        but without leading `''` list item
        and first checks if exists.
        """
        new_titles = []
        # Update the mapping with new news items
        with open(news_file, "r") as f:
            for line in f:
                columns = line.strip().split("\t")
                if len(columns) < 4:
                    continue
                news_id, title = columns[0], columns[3]
                if news_id not in self.news_id2index:
                    self.news_id2index[news_id] = len(self.news_id2index) + 1
                    new_titles.append(title)
        
        # Vectorise and concat new news               
        if not self.vectorizer:
            self.news_titles = tf.concat(
                [self.news_titles, new_titles], axis=0)
        else:
            new_title_vectors = self.vectorizer(new_titles)
            self.news_title_index = tf.concat(
                [self.news_title_index, new_title_vectors], axis=0)     

    def init_behaviors(self, behaviors_file):
        """
        Generator initialising `behaviors` from a behaviors file.

        Behaviors are: `histories, imprs, labels`.

        Parameters:
            behaviors_file (str): The file path to the behaviors data file. Each line in the file should contain tab-separated fields.

        Returns:
            tuple: A tuple histories, imprs, labels where:
                imprs (numpy.ndarray): Padded impression news indices with a fixed length defined by self.max_imps.
                labels (numpy.ndarray): Padded impression labels (converted to integers), using a default of -1 for padding.
                histories (numpy.ndarray): Padded user history indices with a fixed length defined by self.max_hist_len.
                Optional: user_indices (list): A list mapping each user to their unique index as determined by the behaviors file.
        """

        uid2index = {}
        user_indices = []
        with open(behaviors_file, "r") as f:            
            for line in f:
                user_id = line.strip().split("\t")[1]
                if user_id not in uid2index:
                    uid2index[user_id] = len(uid2index) + 1  
                    # Note: end of loop closes context manager
                    # So we need to re-open it.

        with open(behaviors_file, "r") as f: 
            for i, line in enumerate(f):
                columns = line.strip().split("\t")
            
                if len(columns) < 4:
                    continue
                user_id, history, impr = columns[1], columns[3], columns[4]
      
                # history sequences
                history = [self.news_id2index[i] for i in history.split()] or [] 
            
                # impressions sequences
                impr_news = [self.news_id2index[i.split("-")[0]] for i in impr.split()] or []
               
                # label sequences
                label = [int(i.split("-")[1]) for i in impr.split()] or []

                # map user id to index
                uindex = uid2index[user_id] if user_id in uid2index else 0
                user_indices.append(uindex)
       
                yield history, impr_news, label # uid2index, impr_indices   
        self.user_indices = user_indices

    def init_len(self):
        """Computes length of train and validation set.
        Used in combinator with generator (ragged batches)."""
        self.train_size = len(list(self.init_behaviors(self.train)))
        self.steps_per_epoch = max(self.train_size // self.batch_size, 1)
        val_size = len(list(self.init_behaviors(self.dev)))
        self.validation_steps = max(val_size // self.batch_size, 1)    

    def add_padding(self, histories, imprs, labels, pad_len):
        """
        Adds padding to behaviors set. 
        For neg sampling (=if sampling rate `self.K` is set), 
        we reduce padding to 1 + K.
        """        
        histories = pad_sequences(
            histories, self.max_hist_len, 
            padding='pre', truncating='pre')    # take the recent history (most relevant)
        imprs = pad_sequences(
                imprs, pad_len, 
                padding='post', 
                truncating='post')
        labels = pad_sequences(
                labels, pad_len, 
                padding='post', 
                truncating='post', 
                value=-1, 
                dtype=np.float32)
        return histories, imprs, labels
    
    def init_padded_behaviors(self, behaviors_file): 
        """Collects generator for dataset creation"""        
        histories, imprs, labels = self.add_padding(
            *zip(*self.init_behaviors(behaviors_file)), pad_len=self.max_imps)         
        self.train_size = len(histories)
        return histories, imprs, labels    
    
    def get_train_neg(self, K = 4):
        """
        Yields negative sampling sequences for training.
        Can be padded to equal length or in ragged batches.
        """
        self.init_len()
        self.K = K

        # Iterate over get_train
        for histories, imprs, labels in self.init_behaviors(self.train):
            imprs = np.array(imprs)
            labels = np.array(labels)    
            pos_clicks = imprs[labels == 1]
            neg_clicks = imprs[labels == 0] # takes the whole sequence (variable)

            # for every 1 pos sample 4 (K) neg
            for pos in pos_clicks:
                if neg_clicks.shape[0] >= K:
                    neg_samples = np.random.choice(neg_clicks, K, replace=False)
                else:
                    add_same = K - neg_clicks.shape[0]
                    neg_samples = np.concatenate(
                        [neg_clicks, np.random.choice(neg_clicks, add_same)], dtype=np.int32)

                shuffle_like_so = np.random.permutation(K+1)

                new_labels = np.array([1,0,0,0,0], dtype=np.int32)[shuffle_like_so]
                new_imprs = np.concatenate([np.array([pos]), neg_samples], dtype=np.int32)[shuffle_like_so]
              
                # every new sequence gets the full history
                yield histories, new_imprs, new_labels
        

    
    def get_dataset(self, behaviors_file):     
        """
        Collects padded behaviors and loads `tf.data.Dataset` object. 
        """
        histories, imprs, labels = self.init_padded_behaviors(behaviors_file)

        if np.max(imprs) > self.logging_max_idx:
            self.logging_max_idx = np.max(imprs)

        if any(s for s in behaviors_file.as_posix() if ('test', 'dev')):
            self.test_imprs = imprs
            self.test_labels = labels
     
        # Create a dataset from the tensors
        return tf.data.Dataset.from_tensor_slices((histories, imprs, labels)) 
         
    def apply_vectoriser(self, vectorizer):
        """
        Applies the vectorizer
        Loads the `news_title_index` with a lookup table for token_id sequences.
        """
        # Adapt the vectorizer to the titles (i.e. build vocab)
        self.vectorizer = vectorizer
        self.vectorizer.adapt(self.news_titles)

        # Vectorize news titles
        self.news_title_index = self.vectorizer(self.news_titles)
          
    def map_fn(self, history, impr, label):
        """
        Maps user history and impression ids to `news_title_index`, 
        a lookup table for token_id sequences, 
        and returns these and their corresponding labels.
        """
        # Retrieve token sequences for history and impr
        user_history_tokens = tf.gather(self.news_title_index, history)
        candidate_news_tokens = tf.gather(self.news_title_index, impr) 
        return (user_history_tokens, candidate_news_tokens), label

    # I did raw strings as well, but I prefer the lookup layer    
    def map_fn_str(self, history, impr, label):
        raise NotImplementedError
        """
        Alternative mapping.
        Map user history and impression ids to strings, 
        and return these and their corresponding labels.
        """
        user_history_str = ops.take(self.news_titles, history)  # gets news_titles instead of idxs!
        candidate_news_str = ops.take(self.news_titles, impr)  # use ops.take

        # Expand dims at the end
        user_history_str = ops.expand_dims(user_history_str, axis=-1)
        candidate_news_str = ops.expand_dims(candidate_news_str, axis=-1)
        return (user_history_str, candidate_news_str), label    
    # raw_dataset.map(self.map_fn_str, num_parallel_calls=tf.data.AUTOTUNE)

    def get_train_ds(self, vectorizer=None):
        """
        Gets the dataset and optionally applies a `vectorizer`. 
        Maps behaviours to tokens, and returns a batched dataset.        
        """ 
        raw_dataset = self.get_dataset(self.train)

        self.train_size = raw_dataset.cardinality().numpy()
        self.steps_per_epoch = max(self.train_size // self.batch_size, 1)

        if vectorizer:  
            self.apply_vectoriser(vectorizer)
            return raw_dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                .shuffle(buffer_size=self.train_size)\
                .repeat().batch(self.batch_size)\
                    .prefetch(tf.data.AUTOTUNE) 
        
        # print('Mapping to strings!')# .map(self.map_fn_str, num_parallel_calls=tf.data.AUTOTUNE)  
        return raw_dataset.map(lambda x,y,z: ((x,y),z))\
            .shuffle(buffer_size=self.train_size)\
                .repeat().batch(self.batch_size)\
                    .prefetch(tf.data.AUTOTUNE) 
    
    def get_train_ds_negs(self, vectorizer=None, fixed_padding=False):
        """
        Gets the train dataset with negative sampling. 
        Optionally applies a `vectorizer`, and returns a batched dataset.     
        """  
        if fixed_padding:
            pad_len = self.max_imps
        else:
            pad_len = 5

        padded_train_dataset = self.add_padding(
            *zip(*self.get_train_neg()), pad_len)
   
        raw_dataset = tf.data.Dataset.from_tensor_slices(padded_train_dataset)
        
        self.train_size = raw_dataset.cardinality().numpy()
        self.steps_per_epoch = max(self.train_size // self.batch_size, 1)

        if vectorizer:  
            self.apply_vectoriser(vectorizer)
            return raw_dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                .shuffle(buffer_size=self.train_size)\
                .repeat().batch(self.batch_size)\
                    .prefetch(tf.data.AUTOTUNE) 
        
        return raw_dataset.map( # BERT requires an extra dim at the end
            lambda x,y,z: ((ops.expand_dims(x, axis=-1),ops.expand_dims(y, axis=-1)),z)
            ).shuffle(buffer_size=self.train_size)\
                .repeat().padded_batch(self.batch_size)\
                    .prefetch(tf.data.AUTOTUNE)
    
    def get_val(self):
        """
        Gets the `validation` dataset and vectoriser (if given in get_train).

        :Note: Returns the test set if dataloader is initialised with `small`.        
        """          
        raw_dataset = self.get_dataset(self.dev)

        val_size = raw_dataset.cardinality().numpy()
        self.validation_steps = max(val_size // self.batch_size, 1)  

        if self.vectorizer is None:
            return raw_dataset.map(lambda x,y,z: ((x,y),z))\
                .repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE) 
        return raw_dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)\
        .repeat().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    

    def get_test(self):
        """
        Gets the `test` dataset by updating the `news_title_index`.
        New test news (not seen during training) are vectorised with the existing vectoriser
        and appended to self.news_title_index.
        """
        self.update_news(self.test_news) 
        raw_dataset = self.get_dataset(self.test)
        if self.vectorizer is None:
            return raw_dataset.map(lambda x,y,z: ((x,y),z))\
            .padded_batch(self.batch_size).prefetch(tf.data.AUTOTUNE) 
        
        return raw_dataset.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)\
            .padded_batch(self.batch_size).prefetch(tf.data.AUTOTUNE) 
    
    def get_popularity(self):
        """
        Gets the popularity counter. Three seconds.
        
        Returns: tuple (dict, int): item_popularity, total_users
        """
        # if not np.any(self.labels):
        #     raise ValueError('First init behaviours to load label and imprs attributes.')
        from recs.metrics import popularity_count
        import pandas as pd
        _, imprs, labels = zip(*self.init_behaviors(self.train))
        
        # Drop duplicates        
        user_behavior_data = pd.read_csv(self.train, 
                                         sep='\t', header=None, 
                                         names=['user_id', 'news_history'], 
                                         usecols=[1,3])
        
        histories = user_behavior_data.drop_duplicates('user_id').news_history.str.split()
        self.total_users = histories.count()
        histories = [hi for st in histories.tolist() if isinstance(st, list) for hi in st]
        histories = [self.news_id2index[i] for i in histories]       

        self.item_popularity, self.total_counts = popularity_count(imprs, labels, histories)
        del histories, imprs, labels, _
    
    def get_ragged_batch(self, behaviors_file, batch_size=64, vectorizer=None): 
        """Gets a variable sized batch. TBC."""    
        self.init_len()
        if behaviors_file == self.train:
            def gen():
                for histories, imprs, labels in self.get_train_neg():
                    yield histories, imprs, labels  
        else:
            def gen(behaviors_file):
                for histories, imprs, labels in self.init_behaviors(behaviors_file):
                    yield histories, imprs, labels  

        train_ds = tf.data.Dataset.from_generator(gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int64),   # histories
                tf.TensorSpec(shape=(None,), dtype=tf.int64),   # imprs
                tf.TensorSpec(shape=(None,), dtype=tf.int64)    # labels
            ))
        
        if vectorizer:
            self.apply_vectoriser(vectorizer)
            train_ds = train_ds.map(self.map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        else:
            train_ds = train_ds.map(lambda x,y,z: ((x,y),z))
            
        if behaviors_file == self.train:
            train_ds = train_ds.shuffle(buffer_size=self.train_size)
        
        batched_train = train_ds.repeat().padded_batch(
                batch_size=batch_size,
                padded_shapes=(([None], [None]), [None]),
                padding_values=((tf.constant(0, dtype=tf.int64), # TODO
                                 tf.constant(0, dtype=tf.int64)), 
                                 tf.constant(-1, dtype=tf.int64))
            )
        return batched_train.prefetch(tf.data.AUTOTUNE)