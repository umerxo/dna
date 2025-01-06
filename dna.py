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

# encode nucleotides
def encode_sequence(seq):
    return [nucleotide_map.get(base, 4) for base in seq]

encoded_segments = [encode_sequence(seg) for seg in segments]

# introduce synthetic substitutions
def introduce_substitutions(seq, error_rate=0.01):
    seq = seq.copy()
    for i in range(len(seq)):
        if random.random() < error_rate:
            current_base = seq[i]
            possible_bases = [b for b in nucleotide_map.values() if b != current_base]
            seq[i] = random.choice(possible_bases)
    return seq

noisy_segments = [introduce_substitutions(seg) for seg in encoded_segments]

# padding sequences
max_length = 1000
padded_segments = keras.preprocessing.sequence.pad_sequences(encoded_segments, maxlen=max_length, padding='post', value=4)
padded_noisy_segments = keras.preprocessing.sequence.pad_sequences(noisy_segments, maxlen=max_length, padding='post', value=4)

# create labels
labels = np.array(padded_segments != padded_noisy_segments, dtype=int)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_noisy_segments, labels, test_size=0.2, random_state=42)

# design the CNN architecture
model = keras.Sequential([
    layers.Embedding(input_dim=5, output_dim=16, input_length=max_length),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(max_length, activation='sigmoid')
])

# compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# evaluations
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# predict and correct errors
predictions = model.predict(X_test)
predicted_errors = (predictions > 0.5).astype(int)
corrected_sequences = X_test.copy()
corrected_sequences[predicted_errors] = padded_segments[predicted_errors]

# calculate performance metrics
precision = precision_score(y_test.flatten(), predicted_errors.flatten())
recall = recall_score(y_test.flatten(), predicted_errors.flatten())
f1 = f1_score(y_test.flatten(), predicted_errors.flatten())

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
