import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

genome_folder_path = '/content/drive/My Drive/genomes'
if not os.path.exists(genome_folder_path):
    raise FileNotFoundError(f"The specified genome folder path '{genome_folder_path}' doesn't exist")

output_folder_path = '/content/drive/My Drive/genome_fixed'
os.makedirs(output_folder_path, exist_ok=True)

# dictionary to encode nucleotides to integers and back
nucleotide_to_int = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
    'N': 4 
}
int_to_nucleotide = {v: k for k, v in nucleotide_to_int.items()}

# define a fixed sequence length
SEQ_LENGTH = 1000

# custom dataset for genome sequences
class GenomeDataset(Dataset):
    def __init__(self, folder_path, seq_length):
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fa')]
        self.sequences = []
        self.seq_length = seq_length
        self.headers = []
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                header = None
                for line in f:
                    if line.startswith('>'):
                        header = line.strip()
                    else:
                        encoded_sequence = [nucleotide_to_int.get(nuc, 4) for nuc in line.strip()]
                        self.sequences.append(encoded_sequence)
                        self.headers.append(header)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        header = self.headers[idx]
        # pad or truncate the sequence to self.seq_length
        if len(sequence) < self.seq_length:
            sequence = sequence + [4] * (self.seq_length - len(sequence))
        else:
            sequence = sequence[:self.seq_length]
        sequence = np.array(sequence, dtype=np.int64)  # Use int64 for class indices

        # one-hot encode the input sequence
        inputs = np.eye(5)[sequence]  # shape: (seq_length, 5)

        # pad the sequence to length 1024 for reshaping into 32x32 image
        target_length = 1024
        if inputs.shape[0] < target_length:
            pad_size = target_length - inputs.shape[0]
            inputs = np.pad(inputs, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            sequence = np.pad(sequence, (0, pad_size), mode='constant', constant_values=4)
        else:
            inputs = inputs[:target_length]
            sequence = sequence[:target_length]

        # reshape inputs to (height, width, channels)
        height = 32
        width = 32
        inputs = inputs.reshape((height, width, 5))

        # transpose to (channels, height, width) for pytorch
        inputs = inputs.transpose((2, 0, 1))  # Shape: (5, 32, 32)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(sequence, dtype=torch.long), header

# define the cnn model with improvements 
class GenomeCNN(nn.Module):
    def __init__(self):
        super(GenomeCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)

        # compute the size of the output from the conv layers
        self.output_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.output_size, 512)
        self.batch_norm_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.batch_norm_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1024 * 5)  # output logits for each nucleotide at each position

    def _get_conv_output_size(self):
        # create a dummy input with the expected input dimensions
        dummy_input = torch.zeros(1, 5, 32, 32)
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(dummy_input))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        output_size = x.view(1, -1).size(1)
        return output_size

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten appropriately
        x = torch.relu(self.batch_norm_fc1(self.fc1(x)))
        x = torch.relu(self.batch_norm_fc2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 5, 1024)  # reshape to (batch_size, num_classes=5, seq_length=1024)
        return x  # raw logits

# hyperparameters
batch_size = 16
epochs = 15  # can be adjusted
learning_rate = 0.00005  # can be adjusted
weight_decay = 1e-5  # weight decay for l2

# load dataset
dataset = GenomeDataset(genome_folder_path, seq_length=SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenomeCNN().to(device)  # Use gpu if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    model.train()  # set model to training mode
    for i, (inputs, targets, _) in enumerate(dataloader):
        # move inputs and targets to device
        inputs = inputs.to(device)  # shape: (batch_size, 5, 32, 32)
        targets = targets.to(device)  # shape: (batch_size, 1024)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)  # outputs shape: (batch_size, 5, 1024)
        loss = criterion(outputs, targets)  # CrossEntropyLoss expects (batch_size, num_classes, seq_length) and (batch_size, seq_length)
        loss.backward()
        optimizer.step()

        # Print stats
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# find errors and create fixed sequences
model.eval()  # set model to evaluation mode
output_filename = os.path.join(output_folder_path, 'all_fixed_sequences.fa')

# initialize counters for evaluation metrics
total_errors = 0
total_positions = 0
correct_predictions = 0

# for calculating per-class metrics
confusion_matrix = np.zeros((5, 5), dtype=np.int64)  # rows: true labels columns: predicted labels

with torch.no_grad():
    with open(output_filename, 'w') as output_file:
        for i, (inputs, targets, headers) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get the index of the max log-probability
            errors = (predicted != targets)  # boolean tensor of errors
            errors_np = errors.cpu().numpy()
            targets_np = targets.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            # update total errors and positions
            total_errors += errors_np.sum()
            total_positions += targets.numel()
            correct_predictions += (predicted == targets).sum().item()

            # update confusion matrix
            for t, p in zip(targets_np.flatten(), predicted_np.flatten()):
                confusion_matrix[t, p] += 1

            for idx in range(inputs.shape[0]):
                original_sequence = targets_np[idx]
                predicted_sequence = predicted_np[idx]
                header = headers[idx]

                # map back to nucleotide characters
                fixed_sequence = [int_to_nucleotide[nuc] for nuc in predicted_sequence[:SEQ_LENGTH]]  # trim to original SEQ_LENGTH
                fixed_sequence_str = "".join(fixed_sequence)

                # write to a single .fa file
                output_file.write(f"{header}\n")
                output_file.write(fixed_sequence_str + "\n")

    # calculate evaluation metrics
    overall_accuracy = correct_predictions / total_positions * 100
    error_percentage = total_errors / total_positions * 100

    print(f'\nTotal positions analyzed: {total_positions}')
    print(f'Total errors detected: {total_errors}')
    print(f'Overall accuracy: {overall_accuracy:.4f}%')
    print(f'Error percentage: {error_percentage:.4f}%')

    # calculate per-class precision, recall, and F1-score
    precision_per_class = np.zeros(5)
    recall_per_class = np.zeros(5)
    f1_per_class = np.zeros(5)

    for i in range(5):
        TP = confusion_matrix[i, i]
        FP = confusion_matrix[:, i].sum() - TP
        FN = confusion_matrix[i, :].sum() - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_per_class[i] = precision
        recall_per_class[i] = recall
        f1_per_class[i] = f1

    nucleotides = ['A', 'T', 'C', 'G', 'N']
    print('\nPer-class evaluation metrics:')
    for i in range(5):
        print(f"Nucleotide '{nucleotides[i]}': Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1 Score: {f1_per_class[i]:.4f}")

print(f'\nFixed sequences have been saved to {output_filename}')
