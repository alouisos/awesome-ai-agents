import numpy as np

# Set up reproducibility
np.set_printoptions(precision=4, suppress=True)

# Input tokens: "The" = 1, "cat" = 2
tokens = [1, 2]  # Token IDs
seq_len = len(tokens)
import numpy as np

# Set up reproducibility
np.set_printoptions(precision=4, suppress=True)

# Input tokens: "The" = 1, "cat" = 2
tokens = [1, 2]  # Token IDs
seq_len = len(tokens)

# === Embedding matrix (vocab_size x d_model) ===
vocab_size = 5
d_model = 2
embedding_matrix = np.array([
    [0.1, 0.0],   # ID 0
    [0.2, 0.4],   # "The"
    [0.3, 0.1],   # "cat"
    [0.0, 0.5],
    [0.6, 0.2]
])

# === Positional encoding (learned, dummy for now) ===
pos_encoding = np.zeros((seq_len, d_model))  # no effect here, just for structure

# === Input embeddings ===
X = embedding_matrix[tokens] + pos_encoding  # Shape: (2, 2)

# === Self-Attention weights ===
W_Q = np.array([[1.0, 0.0],
                [0.0, 1.0]])

W_K = np.array([[0.5, 0.5],
                [0.5, -0.5]])

W_V = np.array([[1.0, 1.0],
                [0.0, 1.0]])

# === Compute Q, K, V ===
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# === Attention scores ===
scores = Q @ K.T / np.sqrt(d_model)

# === Causal mask ===
mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
masked_scores = scores + mask

# === Softmax ===
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=-1, keepdims=True)

attn_weights = softmax(masked_scores)
attn_output = attn_weights @ V  # Shape: (2, 2)

# === FFN Layer ===
d_ff = 4
W1 = np.array([
    [1.0,  0.5, -0.5,  1.0],
    [0.0, -1.0,  1.0,  0.5]
])  # (2, 4)
b1 = np.array([0.0, 0.1, 0.0, -0.1])  # (4,)

W2 = np.array([
    [0.5, -0.5],
    [-0.5, 0.5],
    [1.0, 1.0],
    [0.0, -1.0]
])  # (4, 2)
b2 = np.array([0.0, 0.0])  # (2,)

# Apply FFN to each token
def relu(x):
    return np.maximum(0, x)

def ffn(x):
    x = x @ W1 + b1
    x = relu(x)
    x = x @ W2 + b2
    return x

ffn_output = np.vstack([ffn(x) for x in attn_output])  # (2, 2)

# === Final output projection to logits ===
W_vocab = np.array([
    [ 0.5,  1.0, -1.0,  0.3,  0.0],
    [-0.5,  0.0,  1.0, -0.2,  0.5]
])  # (2, 5)

logits = ffn_output[-1] @ W_vocab  # Only use last token
probs = softmax(logits)

# === Print results ===
print("Input tokens:", tokens)
print("Embeddings:\n", X)
print("\nQ:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("\nAttention scores:\n", scores)
print("Masked scores:\n", masked_scores)
print("Attention weights:\n", attn_weights)
print("Attention output:\n", attn_output)
print("\nFFN output:\n", ffn_output)
print("\nFinal logits:\n", logits)
print("Softmax probs:\n", probs)
print("\nPredicted token ID:", np.argmax(probs))