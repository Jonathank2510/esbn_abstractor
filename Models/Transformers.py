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

    self.decoder = MultiAttentionDecoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff, num_context=1,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder([x, context])  # (batch_size, target_len, d_model)

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
  

  
# Transformer that uses the Abstractor Module
class TransformerAbstracter(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, use_self_attention=False,
               use_esbn=False, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.abstracter = Abstracter(num_layers=num_layers, d_model=d_model,
                                     num_heads=num_heads, dff=dff,
                                     vocab_size=input_vocab_size, use_esbn=use_esbn,
                                     use_self_attention=use_self_attention, dropout_rate=dropout_rate)
    self.decoder = MultiAttentionDecoder(num_layers=num_layers, d_model=d_model,
                                         num_heads=num_heads, dff=dff,
                                         vocab_size=target_vocab_size,
                                         num_context=2, dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    context, x  = inputs

    context_symbols = self.abstracter(context)
    context = self.encoder(context)
     
    x = self.decoder([x, context_symbols, context])

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