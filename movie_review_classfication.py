import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
import tensorflow_datasets as tfds

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
training_set, test_set = dataset['train'], dataset['test']
encoder = info.features['text'].encoder 
#encoder reduces dimensional representation of set of words
buffer_size = 10000
batch_size = 64
padded_shapes = ([None],())

training_set = training_set.shuffle(buffer_size).padded_batch(batch_size, 
                                                              padded_shapes=padded_shapes)
test_set = test_set.shuffle(buffer_size).padded_batch(batch_size, 
                                                              padded_shapes=padded_shapes)
# Sequential model 
'''
model = keras.Sequential([layers.Embedding(encoder.vocab_size, 64),
                          layers.Bidirectional(layers.LSTM(64)),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(1, activation='sigmoid')])

model.compile(loss= 'binary_crossentropy',
              optimizer=k.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(training_set, 
                    epochs = 1, 
                    validation_data = test_set,
                    validation_steps = 30)
'''

def pad_to_size(vector, size):
  zeros = [0]*(size-len(vector))
  vector.extend(zeros)
  return vector

def sample_predict(sentence, pad):
  encoder_sample_pred_text = encoder.encode(sentence)
  if pad:
    encoder_sample_pred_text = pad_to_size(encoder_sample_pred_text, 64)
  encoder_sample_pred_text = tf.cast(encoder_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoder_sample_pred_text, 0))
  return predictions

'''
sample_text = ('The movie was awesome. The acting was incredible')
predictions = sample_predict(sample_text, pad=True model=model)*100
print('this should be a positive review')
'''

model = keras.Sequential([layers.Embedding(encoder.vocab_size, 64),
                          layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                          layers.Bidirectional(layers.LSTM(32)),
                          layers.Dense(64, activation='relu'),
                          layers.Dropout(0.5),
                          layers.Dense(1, activation='sigmoid')])

model.compile(loss= 'binary_crossentropy',
              optimizer=k.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(training_set, 
                    epochs = 4, 
                    validation_data = test_set,
                    validation_steps = 30)

sample_text = ('The movie was awesome. The acting was incredible, in general it is very good way to spend time')
predictions = sample_predict(sample_text, pad=True)*100
print('this should be a positive review')

sample_text = ('The movie was horrible. The acting was bad, in general it was a waste of time')
predictions = sample_predict(sample_text, pad=True)*100
print('this should be a negative review')