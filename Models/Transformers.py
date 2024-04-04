import tensorflow as tf
from Models.TransformerModules import *

# Basic Transformer
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output
    return logits
  


class ESBNTransformerSCA(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)
    
    self.symbolic_decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=target_vocab_size,
                                    dropout_rate=dropout_rate)
    
    self.esbn_encoder = ESBNEncoderCrossAttention(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=input_vocab_size,
                                    dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    context, x  = inputs
    context_symbols = self.esbn_encoder(context)
    context = self.encoder(context)  

    # Use cross-attention with ESBN - generated symbols
    s = self.symbolic_decoder(x, context_symbols) 
    # Use cross-attention with usual encoder
    x = self.decoder(x, context)  

    # Concatenate information streams
    x = tf.concat([x, s], 2)

    # Final linear layer output.
    logits = self.final_layer(x)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits


class ESBNTransformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)
    
    self.symbolic_decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=target_vocab_size,
                                    dropout_rate=dropout_rate)
    
    self.esbn_encoder = ESBNEncoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=input_vocab_size,
                                    dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    context, x  = inputs
    context_symbols = self.esbn_encoder(context)
    context = self.encoder(context)  

    # Use cross-attention with ESBN - generated symbols
    s = self.symbolic_decoder(x, context_symbols) 
    # Use cross-attention with usual encoder
    x = self.decoder(x, context)  

    # Concatenate information streams
    x = tf.concat([x, s], 2)

    # Final linear layer output.
    logits = self.final_layer(x)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits