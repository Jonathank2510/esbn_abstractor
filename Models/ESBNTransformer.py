import tensorflow as tf
from Models.ESBNLayer import ESBNLayer
from Models.TransformerModules import *

# ESBNEncoder converts to sequence of symbols before encoding
class ESBNEncoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.esbn_layer = ESBNLayer(key_size=d_model, hidden_size=32, return_memory=True)

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Create symbols
    lstm_out, (symbols, values) = self.esbn_layer(x)

    # Add dropout.
    symbols = self.dropout(symbols)

    for i in range(self.num_layers):
      symbols = self.enc_layers[i](symbols)

    return symbols  # Shape `(batch_size, seq_len, d_model)`.
  

# ESBNDecoder converts to sequence of symols before decoding
class ESBNDecoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.esbn_layer = ESBNLayer(key_size=d_model, hidden_size=32, return_memory=True)

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    # Create symbols
    lstm_out, (symbols, values) = self.esbn_layer(x)

    symbols = self.dropout(symbols)

    for i in range(self.num_layers):
      symbols  = self.dec_layers[i](symbols, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return symbols

  

class ESBNTransformer2(tf.keras.Model):
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
    
    self.esbn_encoder = ESBNEncoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=input_vocab_size,
                                    dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context_symbols = self.esbn_encoder(context)
    context = self.encoder(context)  # (batch_size, context_len, d_model)

    s = self.decoder(x, context_symbols) 
    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    x = tf.concat([x, s], 2)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
  



class ESBNTransformer1(tf.keras.Model):
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
    
    self.esbn_encoder = ESBNEncoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=input_vocab_size,
                                    dropout_rate=dropout_rate)
    
    self.esbn_decoder = ESBNDecoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, dff=dff,
                                    vocab_size=target_vocab_size,
                                    dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # Ordinary transformer component
    context, x  = inputs
    context_ordinary = self.encoder(context)  # (batch_size, context_len, d_model)
    x_ordinary = self.decoder(x, context_ordinary)  # (batch_size, target_len, d_model)

    # Symbolic transformer component
    context_symbolic = self.esbn_encoder(context)
    x_symbolic = self.esbn_decoder(x, context_symbolic)

    # Concatenate ordinary and symbolic component
    x = tf.concat([x_ordinary, x_symbolic], 2)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits