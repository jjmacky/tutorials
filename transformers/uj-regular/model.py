import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model # d_model is dimension of embedding vector
        self.vocab_size = vocab_size # The number of words in our vocabulary "Dictonary"
        self.embedding = nn.Embedding(vocab_size, d_model) # Initated randomly, learned along with our model weights via back propogation

    # We multiply by square root of d_model to ensure embeddings are large enough
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # Max length of input sequence (a sentence or passage of text)
        self.dropout = nn.Dropout(dropout)

        # Initialize a matrix of shape (seq_len, d_model)
        # Each token is an element in pe with an embedding vector of size d_model (e.g., 512)
        # pe = position * div_term where construct the div term so that it can be multiplied rather than divided
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        # arange creates a 1D tensor of size (end - start) / step
        # unsqueeze adds an extra dimension of size 1 at the specified location
        # This creates a column vector becaue arange is shape [N]; adding dimension in location 1 creates [N, 1]
        position = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)

        # div_term = 1 / 10000^(2i/d_model).
        # Use the log definition for numerical stability using a^b = exp(b * log(a))
        div_term = torch.exp(torch.arange(0, d_model, 1).float() * (-math.log(10000.0) / d_model))

        # Now we construct pe using the previously calculated position and div_terms
        # Use alternating sin and cosine to provide rich, meaningful position information in each embedding.
        # The period is inversely proportional to the div_term, which is a function of the index i in the d_model embedding.
        # Specifically, div_term = 1 / 10000^(2i/d_model).
        # When i is small, the div_term is large, resulting in a short period (high frequency).
        # When i is large, the div_term is small, resulting in a long period (low frequency).
        # Short periods encode fine-grained position information, such as how closely spaced words relate to each other in the text passage.
        # Long periods encode coarse-grained position information, such as how distantly spaced words relate to each other in the text passage.
        # Because the period varies with i, and i ranges from 0 to d_model - 1, the encoding captures both close and distant textual relationships in the input text.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Becomes a tensor of (batch_size, seq_len, d_model)

        self.register_buffer('pe', pe) # Saves state that is not explicitly a model parameter (another example would be a running average)

        ################################### Aside on torch slicing ###################################
        #   PyTorch tensor slicing and indexing follows the general pattern: [dim 0, dim 1, dim 2, etc.]
        #   Each dim follows [start:stop:step] or [start:stop] which implicitly sets step = 1
        #   Examples for a 2D matrix: mat = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        #       (A) mat[:2, 1:]   # Rows 0 and 1, columns 1 and 2 → [[1, 2], [4, 5]]
        #       (B) mat[::2, :]   # Every second row, all columns → [[0, 1, 2], [6, 7, 8]]
        #       (c) mat[::-1, 0]  # Reverse rows, column 0 → [6, 3, 0]
        #   pe[:, 0::2] means, we select every token in seq_len, and select every second index of the embedding vector of size d_model
        ##############################################################################################

        ################################### Aside on tensor shapes ###################################
        #   By default a PyTorch tensor has shape [N] and is a 1D tensor
        #   A 1D tensor does not have rows or columns, it's a sequence of numbers
        #   unsqueeze(0) turns it into a 2D tensor of size [1, N], aka a row vector with N columns
        #   unsqueeze(1) turns it into a 2D tensor of size [N, 1], aka a column vector with N rows
        #   A tensor shape of [3, 10] is 3 rows and 10 columns
        #   You can also create a 0D scalar: torch.tensor(42). This is just a number like we learned in elementary school.
        #   All of this maps to traditional linear algebra notions:
        #       - A 1D vector is a list of numbers (usually waiting to be transformed into a row or column vector for multiplication)
        #       - A row or column vector is like a matrix with deminsion 1 along either the row or column (e.g., a 1 x 8 matrix is a column vector)
        ##############################################################################################

    # Add positional encoding to every token in the input sequence
    def forward(self, x):
        # Recall that positional encoding (pe) has shape (1, seq_len, d_model)
        # To match the input x, which may be shorter than seq_len, we slice pe to [:, :x.shape[1], :]
        # x has shape (batch_size, num_tokens, d_model), where:
        #   - batch_size is x.shape[0], allowing us to process multiple sequences in parallel
        #   - num_tokens is x.shape[1], the number of tokens in each input sequence
        #   - d_model is the size of the embedding vector for each token
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
    ################################### Aside on Dropout ###################################
    #   Dropout randomly sets some elements of the embedding vector (d_model) to 0.
    #   This helps prevent overfitting by encouraging the model to learn multiple pathways 
    #   for next-token prediction, rather than relying too heavily on specific dimensions 
    #   of the embedding vector.
    #
    #   The dropout rate p is a value between 0 and 1. After dropping values, the remaining 
    #   elements are scaled by 1 / (1 - p) to maintain the expected sum of the tensor.
    #
    #   Example: If x consists of two tokens with an embedding vector of length 3:
    #       x = [[0.8, 0.6, 0.3],
    #        [0.5, 0.2, 0.7]]
    #   With a dropout rate of 0.5, one possible result is:
    #       x = [[1.6, 0.0, 0.6],  # Elements dropped: 0.6 (scaled remaining by 2)
    #        [0.0, 0.4, 0.0]]  # Elements dropped: 0.5 and 0.7
    ########################################################################################


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) # using nn.Parameter automatically sets requires_grad=True so the value can be learned during back propagation
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # We normalize x across the feature dimension (the embedding of size d_model)
        # This insures that each token's features are on a comparable scale
        # After normalization we add two learnable parameters: gamma and beta (or bias)
        # Gamma allows the function to stretch or shrink and the bias can shift them up or down
        # Recall x has size x has shape (1, num_tokens, d_model).
        # We use -1 to indicate we sum along the last dimension, which is d_model
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.bias

class  FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        # W1, W2 and b1, b2 will be randomly initalized by default
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1 (bias is True by default)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and b2 (bias is True by default)

    def forward(self, x):
        # x is of size (batch_size, seq_len, d_model) 
        xW1 = self.linear_1(x) #  We left multiple x by W1: (batch_size, seq_len, d_model) × (d_model x dff) = (batch_size, seq_len, d_ff)
        H = self.dropout(torch.relu(xW1)) # Apply ReLU and dropout
        return self.linear_2(H) # We left multiply by H by W2: (batch_size, seq_len, d_ff) × (d_ff, d_model) = (batch_size, seq_len, d_model)

    ########################### Aside on This Feed Forward Layer ###########################
    #   Multiplying x by W1 expands the embedding dimension from d_model to d_ff 
    #   (2048 in the original Transformer paper).
    #
    #   Next, we apply ReLU to:
    #       1. Allow the model to learn non-linear transformations.
    #       2. Promote sparse representations, which can be more computationally efficient 
    #          and help reduce overfitting.
    #
    #   After ReLU, we apply dropout to further reduce overfitting.
    #
    #   Finally, we project back down to the original embedding dimension (d_model), 
    #   so the output can be used by subsequent layers (e.g., the attention layer).
    ########################################################################################

    ################################## Aside on nn.Linear ##################################
    #   The actual matrix multiplication in the feed-forward layer is an affine transform:
    #       xW^T + b
    #   However, we specify nn.Linear as:
    #       nn.Linear(input_feature_dim, output_feature_dim)
    #
    #   This means we pass arguments to nn.Linear *as if* we were directly multiplying x and W 
    #   without a transpose. For example, if x has shape (batch_size, seq_len, d_model), 
    #   and we do:
    #       nn.Linear(d_model, d_ff)
    #   then by standard matrix multiplication, (batch_size, seq_len, d_model) × (d_model, d_ff)
    #   yields (batch_size, seq_len, d_ff).
    #
    #   Note that INTERNALLY, PyTorch stores the matrix in the opposite order: (output_dim, input_dim).
    #   So if we say nn.Linear(d_model, d_ff), the matrix is physically stored with shape (d_ff, d_model).
    #
    #   You might assume that means we incur an extra matrix transpose at runtime (since we want 
    #   a conceptual shape of (d_model, d_ff), but it's stored as (d_ff, d_model)). However, PyTorch 
    #   avoids an expensive copy by using:
    #       - storage_offset: A pointer to where the “start” of the tensor is in memory.
    #       - stride: The step size used to move through memory along each dimension.
    #   Via these techniques PyTorch can easily jump to any cell in the matrix. 
    #
    #   By manipulating strides (and possibly using BLAS or cuBLAS transpose flags under the hood), 
    #   PyTorch can interpret the stored (d_ff, d_model) data as though it were (d_model, d_ff) 
    #   without physically rearranging anything. This aligns with its row-major format (where each row 
    #   is contiguous in memory) and ensures efficient matrix multiplication calls without an explicit transpose.
    ########################################################################################


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float): # h is the number of attention heads, 8 in the original paper
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Perform floor division so we store an integer value and not a float

        self.wq = nn.Linear(d_model, d_model) # Wq
        self.wk = nn.Linear(d_model, d_model) # Wk
        self.wv = nn.Linear(d_model, d_model) # Wv
        self.wo = nn.Linear(d_model, d_model) # Wo (note that dv × h = d_model)

