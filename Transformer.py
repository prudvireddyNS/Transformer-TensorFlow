import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def call(self):
        even_i = tf.range(0, self.d_model, 2, dtype=tf.float32)
        denominator = tf.pow(10000, even_i / self.d_model)
        position = tf.reshape(tf.range(self.max_sequence_length, dtype=tf.float32), (self.max_sequence_length, 1))
        even_PE = tf.sin(position/ denominator)
        odd_PE = tf.cos(position/ denominator)
        stacked = tf.stack([even_PE, odd_PE], axis = 2)
        PE = tf.reshape(stacked, (self.max_sequence_length, -1))
        return PE


class SentenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_length, d_model, language_to_ix, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super(SentenceEmbedding, self).__init__()
        self.vocab_size = len(language_to_ix)
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.language_to_ix = language_to_ix
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.dropout = tf.keras.layers.Dropout(0.1)

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        def tokenize(sentence, start_token =True, end_token=True):
            sentence_word_indices = [self.language_to_ix[ch] for ch in list(sentence)]
            if start_token:
                sentence_word_indices.insert(0, self.language_to_ix[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_ix[self.END_TOKEN])
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_ix[self.PADDING_TOKEN])
            return sentence_word_indices
        tokenized = []
        for sentence in batch:
            tokenized.append(tokenize(sentence, self.START_TOKEN, self.END_TOKEN))
        tokenized = tf.stack(tokenized)
        return tokenized
    
    def call(self, x,start_token, end_token):
        x = self.batch_tokenize(x)
        x = self.embedding(x)
        pos = self.position_encoder.call()
        out = self.dropout( x + pos )
        return out 
        

class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, ffn_hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = tf.keras.layers.Dense(ffn_hidden, activation='relu')
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(drop_prob)

    def call(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        out = self.linear2(x)
        return out
        

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, parameter_shape,eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.parameter_shape = parameter_shape
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(parameter_shape))
        self.beta = tf.Variable(tf.zeros(parameter_shape))

    def call(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = tf.reduce_mean(inputs, axis = dims, keepdims=True)
        var = tf.reduce_mean((inputs - mean)**2, axis=dims, keepdims=True)
        std = tf.math.sqrt(var + self.eps)
        inputs = (inputs - mean) / std
        out = self.gamma * inputs + self.beta
        return out
        

def scaled_dot_product(q, k, v, mask = None):
    d_k = k.shape[-1]
    scaled = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
    if mask is not None:
        mask_expanded = tf.expand_dims(mask, axis=1)
        scaled += mask_expanded
    attention = tf.keras.activations.softmax(scaled, axis = -1)
    values = tf.matmul(attention, v)
    return attention, values
    

class MultiheadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, 
                d_model, 
                num_heads):
        super(MultiheadCrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = tf.keras.layers.Dense(2 * d_model)
        self.q_layer = tf.keras.layers.Dense(d_model)
        self.linear_layer = tf.keras.layers.Dense(d_model)

    def call(self, x, y, mask): # x = 30 x 200 x 512
        batch_size, seq_len, d_model = x.shape
        q = self.q_layer(y)
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, perm=[0,2,1,3])
        kv = self.kv_layer(x) # 30 x 200 x 512*3
        kv = tf.reshape(kv, [batch_size, seq_len, self.num_heads, 2 * self.head_dim])
        kv = tf.transpose(kv, perm=[0,2,1,3])
        k, v = tf.split(kv, num_or_size_splits=2, axis = -1)
        attention, values = scaled_dot_product(q, k, v, mask)
        values = tf.reshape(values, [batch_size, seq_len, d_model])
        out = self.linear_layer(values)
        return out
        

class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, 
                d_model, 
                num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_modle = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = tf.keras.layers.Dense(3 * d_model)
        self.linear_layer = tf.keras.layers.Dense(d_model)

    def call(self, x, mask): # x = 30 x 200 x 512
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_layer(x) # 30 x 200 x 512*3
        qkv = tf.reshape(qkv, (batch_size, seq_len, self.num_heads, 3 * self.head_dim))
        qkv = tf.transpose(qkv, perm=[0,2,1,3])
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis = -1)
        attention, values = scaled_dot_product(q, k, v, mask)
        values = tf.reshape(values, [batch_size, seq_len, d_model])
        out = self.linear_layer(values)
        return out
        

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                d_model,
                num_heads,
                ffn_hidden,
                drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNormalization([d_model])
        self.dropout1 = tf.keras.layers.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.layer_norm2 = LayerNormalization([d_model])
        self.dropout2 = tf.keras.layers.Dropout(drop_prob)

    def call(self, x, mask):
        _x = x
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + _x)
        return x
        

class SequentialEncoder(tf.keras.Sequential):
    def __init__(self, layers):
        super(SequentialEncoder, self).__init__()
        self.layers_ = layers

    def call(self, x, mask):
        for layer in self.layers_:
            x = layer(x, mask)
        return x
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, 
                 d_model, 
                 num_heads, 
                 ffn_hidden, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_ix,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super(Encoder, self).__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_ix, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers_ = SequentialEncoder([EncoderLayer(d_model, num_heads, ffn_hidden, drop_prob) for _ in range(num_layers)])

    def call(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers_(x, self_attention_mask)
        return x
        

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                d_model,
                num_heads,
                ffn_hidden,
                drop_prob
                ):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNormalization([d_model])
        self.dropout1 = tf.keras.layers.Dropout(drop_prob)

        self.cross_attention = MultiheadCrossAttention(d_model, num_heads)
        self.layer_norm2 = LayerNormalization([d_model])
        self.dropout2 = tf.keras.layers.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.layer_norm3 = LayerNormalization([d_model])
        self.dropout3 = tf.keras.layers.Dropout(drop_prob)

    def call(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y
        y = self.self_attention(y, self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y
        y = self.cross_attention(x, y, cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)

        return y
        

class SequentialDecoder(tf.keras.Sequential):
    def __init__(self, layers):
        super(SequentialDecoder, self).__init__()
        self.layers_ = layers

    def call(self, x, y, self_attention_mask, cross_attention_mask):
        for layer in self.layers_:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                d_model,
                num_heads,
                ffn_hidden,
                drop_prob,
                num_layers,
                max_sequence_length,
                language_to_ix,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN
                ):
        super(Decoder, self).__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_ix, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers_ = SequentialDecoder([DecoderLayer(d_model, num_heads, ffn_hidden, drop_prob) for _ in range(num_layers)])

    def call(self,
            x,
            y,
            self_attention_mask,
            cross_attention_mask,
            start_token,
            end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        out = self.layers_(x, y, self_attention_mask, cross_attention_mask)
        return out


class Transformer(tf.keras.models.Model):
    def __init__(self, 
                d_model,
                num_heads,
                ffn_hidden,
                drop_out,
                num_layers,
                max_sequence_length,
                te_vocab_size,
                eng_to_ix,
                tel_to_ix,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, ffn_hidden, drop_out, num_layers, max_sequence_length,eng_to_ix, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, num_heads, ffn_hidden, drop_out, num_layers, max_sequence_length,tel_to_ix, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = tf.keras.layers.Dense(te_vocab_size)

    def call(self, 
             x, 
             y, 
            encoder_self_attention_mask=None,
            decoder_self_attention_mask=None,
            decoder_cross_attention_mask=None,
             enc_start_token=None,
            enc_end_token=None,
            dec_start_token=None,
            dec_end_token=None
            ):
        x = self.encoder(x, encoder_self_attention_mask, start_token = enc_start_token, end_token = enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token = dec_start_token, end_token = dec_end_token)
        out = self.linear(out)
        return out