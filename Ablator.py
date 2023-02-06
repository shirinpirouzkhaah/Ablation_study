#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: shirizlw
"""

import opennmt 

"""
Inputters of opennmt.models.Transformer are numerical/ token representation. To feed AST into this model, 
AST should be transformed into a sequence of token representations such as tokenized source code or 
a sequence of AST nodes. This sequence can then be fed into the model using the TextInputter, 
SequenceRecordInputter, or other inputters available in OpenNMT. 
"""

class MyCustomTransformer(opennmt.models.Transformer): 
    def __init__(self):
        super().__init__(
            source_inputter=opennmt.inputters.WordEmbedder(256),
            target_inputter=opennmt.inputters.WordEmbedder(256),
            num_layers=[1, 2],
            num_units=256,
            num_heads=8,
            ffn_inner_dim=256,
            dropout=0.2798,
            attention_dropout=0.1873,
            ffn_dropout=0.2134)


model = MyCustomTransformer

"""
The components of opennmt.models.Transformer are:

Inputter: Responsible for encoding the input sequence into a tensor. In opennmt.models.Transformer, it is specified by the source_inputter and target_inputter parameters.

CharConvEmbedder
CharEmbedder
CharRNNEmbedder
ExampleInputter
ExampleInputterAdapter
Inputter
MixedInputter
MultiInputter
ParallelInputter
SequenceRecordInputter
TextInputter
WordEmbedder ****************8
add_sequence_controls
create_sequence_records
load_pretrained_embeddings
write_sequence_record

Embedding layer: Maps the input words into a continuous vector space.

Multi-head attention mechanism: Implements the multi-head attention mechanism, which allows the model to attend to multiple parts of the input sequence simultaneously.

Feed-forward network (FFN): A two-layer neural network that transforms the input into a lower-dimensional representation.

Position-wise fully connected layer: Applies a linear transformation to each position in the sequence.

Layer normalization: Normalizes the activations of each layer to ensure that the values remain within a reasonable range.

Dropout: Randomly drops out activations during training to prevent overfitting.

Encoder and decoder: Implement the main architecture of the Transformer model, including the multi-head attention mechanism, feed-forward network, and residual connections.

Output layer: Produces the final prediction.
"""
"""
hyperparameters in opennmt.models.Transformer are num_layers, num_units, num_heads, ffn_inner_dim, dropout, attention_dropout, and ffn_dropout.
"""

# class MyCustomTransformerAblator(MyCustomTransformer):
#     def __init__(self, original_model, reduce_layers=False, reduce_units=False, remove_heads=False, 
#                  reduce_ffn_inner_dim=False, increase_dropout=False, increase_attention_dropout=False,
#                  increase_ffn_dropout=False, remove_position_wise_fc=False, remove_layer_norm=False,
#                  remove_output_layer=False, use_char_embedding=False, use_word_embedding=False):
#         super().__init__()
#         self.original_model = original_model
#         self.reduce_layers = reduce_layers
#         self.reduce_units = reduce_units
#         self.remove_heads = remove_heads
#         self.reduce_ffn_inner_dim = reduce_ffn_inner_dim
#         self.increase_dropout = increase_dropout
#         self.increase_attention_dropout = increase_attention_dropout
#         self.increase_ffn_dropout = increase_ffn_dropout
#         self.remove_position_wise_fc = remove_position_wise_fc
#         self.remove_layer_norm = remove_layer_norm
#         self.remove_output_layer = remove_output_layer
#         self.use_char_embedding = use_char_embedding
#         self.use_word_embedding = use_word_embedding

#     def forward(self, source, target):
#         if self.reduce_layers:
#             self.original_model.num_layers = [1, 2]
#         if self.reduce_units:
#             self.original_model.num_units = 128
#         if self.remove_heads:
#             self.original_model.num_heads = 1
#         if self.reduce_ffn_inner_dim:
#             self.original_model.ffn_inner_dim = 128
#         if self.increase_dropout:
#             self.original_model.dropout = 0.3
#         if self.increase_attention_dropout:
#             self.original_model.attention_dropout = 0.2
#         if self.increase_ffn_dropout:
#             self.original_model.ffn_dropout = 0.25
#         if self.remove_position_wise_fc:
#             self.original_model.position_wise_fc = False
#         if self.remove_layer_norm:
#             self.original_model.layer_norm = False
#         if self.remove_output_layer:
#             self.original_model.output_layer = False
#         if self.use_char_embedding:
#             self.original_model.embedder = opennmt.inputters.CharConvEmbedder()
#         if self.use_word_embedding:
#             self.original_model.embedder = opennmt.inputters.WordEmbedder()
            
#         return self.original_model(source, target)
    
    

# original_model = MyCustomTransformer

# ablated_model = MyCustomTransformerAblator(original_model, reduce_layers=True, reduce_units=True, 
#                                             remove_heads=True, reduce_ffn_inner_dim=True, 
#                                             increase_dropout=True, increase_attention_dropout=True, 
#                                             increase_ffn_dropout=True, remove_position_wise_fc=True, remove_layer_norm=False,
#                                             remove_output_layer=True, use_char_embedding=True, 
#                                             use_word_embedding=True)

# source = # some input source
# target = # some input target
# output = ablated_model(source, target)


#ablated_model_output = ablated_model(source, target)


"""
YAML file ablation study :

    
optimizer: different optimization algorithms like SGD, Adadelta, etc. to see which one 
gives the best results.

learning_rate:  different learning rates to see how it affects the training process. (done by Rosalia)

beam_width: different beam widths to see how it affects the translation quality.

num_hypotheses:  different values for the number of hypotheses to see how it affects 
the translation quality.

batch_size: different batch sizes to see how it affects the training speed and convergence.

max_step:  different values for the maximum number of steps to see how it affects 
the training time and convergence.
"""








