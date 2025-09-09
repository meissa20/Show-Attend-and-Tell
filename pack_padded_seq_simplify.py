import torch
from torch.nn.utils.rnn import pack_padded_sequence

sequences = torch.tensor([
    [4, 5, 0, 0, 0],   # length 2
    [1, 2, 3, 0, 0]   # length 3
])

lengths = [2, 3]  # <-- not sorted (should be [4,3,2])

packed = pack_padded_sequence(sequences, lengths, batch_first=True,  enforce_sorted=False)
print (packed)

sequences = torch.tensor([
    
[  [0.2, 0.1, 2.2, 0.5],   # step1
   [0.3, 0.4, 0.2, 2.5],   # step2
   [0.9, 0.8, 0.7, 0.6],   # pad
   [0.9, 0.8, 0.7, 0.6],   # pad
   [0.9, 0.8, 0.7, 0.6] ] , # pad
    
[  [2.0, 1.0, 0.1, 0.2],   # step1
   [0.1, 0.5, 2.5, 0.3],   # step2
   [0.3, 0.2, 0.1, 3.0],   # step3
   [0.9, 0.8, 0.7, 0.6],   # pad
   [0.9, 0.8, 0.7, 0.6] ]  # pad
 
])


packed = pack_padded_sequence(sequences, lengths, batch_first=True,  enforce_sorted=False)
print (packed)

