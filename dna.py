import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from google.colab import drive
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

drive.mount('/content/drive')
genome_folder_path = '/content/drive/My Drive/genomes'

chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']

nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
int_to_nucleotide = {v: k for k, v in nucleotide_map.items()}

def read_fasta(file_path):
    sequences = []
    try:
        for record in SeqIO.parse(file_path, 'fasta'):
            sequences.append(str(record.seq).upper())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return sequences

sequences = {}
for chrom in chromosomes:
    file_path = os.path.join(genome_folder_path, f'chr{chrom}.fa')
    seq_list = read_fasta(file_path)
    if seq_list:
        sequences[chrom] = seq_list[0]
    else:
        print(f"No sequences found for chromosome {chrom}")

window_size = 1000
segments = []
for chrom in chromosomes:
    if chrom in sequences:
        seq = sequences[chrom]
        for i in range(0, len(seq), window_size):
            segment = seq[i:i+window_size]
            if len(segment) == window_size:
                segments.append(segment)

def encode_sequence(seq):
    return [nucleotide_map.get(base, 4) for base in seq]

encoded_segments = [encode_sequence(seg) for seg in segments]

def introduce_substitutions(seq, error_rate=0.01):
    seq = seq.copy()
    for i in range(len(seq)):
        if random.random() < error_rate:
            current_base = seq[i]
            possible_bases = [b for b in nucleotide_map.values() if b != current_base]
            seq[i] = random.choice(possible_bases)
    return seq

noisy_segments = [introduce_substitutions(seg) for seg in encoded_segments]

max_length = 1000
padded_segments = keras.preprocessing.sequence.pad_sequences(
    encoded_segments, maxlen=max_length, padding='post', value=4
)
padded_noisy_segments = keras.preprocessing.sequence.pad_sequences(
    noisy_segments, maxlen=max_length, padding='post', value=4
)

labels = (padded_segments != padded_noisy_segments).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    padded_noisy_segments, labels, test_size=0.2, random_state=42
)

inputs = keras.Input(shape=(max_length,), dtype='int32')
x = layers.Embedding(input_dim=5, output_dim=16, input_length=max_length)(inputs)

x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=2)(x)  # sequence length becomes 500
x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)

x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x) 

x = layers.UpSampling1D(size=2)(x)  

x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
outputs = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)
outputs = layers.Reshape((max_length,))(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f'\nTest metrics:\n- Loss:      {loss:.4f}\n- Accuracy:  {accuracy:.4f}\n- Precision: {precision:.4f}\n- Recall:    {recall:.4f}')

predictions = model.predict(X_test)
predicted_errors = (predictions > 0.5).astype(int)

corrected_sequences = X_test.copy()
corrected_sequences[predicted_errors == 1] = padded_segments[predicted_errors == 1]

# Calculate overall performance metrics (flattened across all positions).
precision_overall = precision_score(y_test.flatten(), predicted_errors.flatten())
recall_overall = recall_score(y_test.flatten(), predicted_errors.flatten())
f1_overall = f1_score(y_test.flatten(), predicted_errors.flatten())

print(f'\nOverall metrics (per base):\n- Precision: {precision_overall:.4f}\n- Recall:    {recall_overall:.4f}\n- F1 Score:  {f1_overall:.4f}')
