

# PICOGPT: The Tiniest Transformer Every for Intuition 
[see full article here for full description of the intuition of the PicoGPT](https://www.linkedin.com/article/edit/7312375128015691776/)

Machine learning transformers have revolutionized the way computers understand language, powering tools like chatbots and translation apps.

But what exactly are they, and how do they work?

In this article, we’ll explain the core idea behind transformers using a simple toy example—so simple, that you will immediately grasp how they work.

Whether you're new to machine learning or just curious, this playful intro will help you grasp the magic behind transformers. This toy example is for **decoders only**, which is the structure of ChatGPT-like assistants.

---

## Tokenization: Numbers, Not Words

Let’s say we want to predict the next word after the sequence:

> "The cat"

How would we teach a computer to do that?

Computers don't understand words but are great with numbers. So, we first represent words as numbers using **tokenizers** — a fancy way to describe a table that maps each word in our vocabulary to a number.

Example vocabulary:

- "The" → 1
- "cat" → 2
- "sat" → 3
- Two more words → 4, 5

So, the phrase:

> "The cat" → `[1, 2]`

This list doesn't contain deep meaning. Next, we define how big the model is—how many input tokens it sees.

Assume:
- Model input context = 2
- Vocabulary size = 5

So, the embedding matrix is `5 × 2` (vocab size × embedding dim).

```python
Embedding matrix E = [
  [0.1, 0.0],   # ID 0
  [0.2, 0.4],   # ID 1 → "The"
  [0.3, 0.1],   # ID 2 → "cat"
  [0.0, 0.5],   # ID 3
  [0.6, 0.2]    # ID 4
]
```

So:
- "The" → `[0.2, 0.4]`
- "cat" → `[0.3, 0.1]`

Embedding matrix input X:
```python
X_embedding = [
  [0.2, 0.4],
  [0.3, 0.1]
]  # Shape: 2 × 2
```

---

## Positional Encoding

We add information about word position using a second matrix:

```python
X_position = [
  [1, 0],
  [0, 1]
]
```

The real models use sinusoidal or learned encodings, but we simplify.

So, the full input becomes:
```python
X = X_embedding + X_position
```

But for now, we proceed with just `X_embedding`.

---

## Self-Attention: Q, K, V

To learn which word relates to what, we use **self-attention**:

We project each word embedding into three vectors:
- Query (Q): What this word is asking about
- Key (K): What it offers
- Value (V): The actual info

We use weight matrices to compute them:

```python
W_Q = [[1.0, 0.0], [0.0, 1.0]]
W_K = [[0.5, 0.5], [0.5, -0.5]]
W_V = [[1.0, 1.0], [0.0, 1.0]]
```

Multiply X with W:
```python
Q = X @ W_Q = [[0.2, 0.4], [0.3, 0.1]]
K = X @ W_K = [[0.3, -0.1], [0.2, 0.1]]
V = X @ W_V = [[0.2, 0.6], [0.3, 0.4]]
```

---

## Attention Scores

We calculate scores by:
```python
scores = Q @ K.T
```

Then divide by `sqrt(d_model)` and apply a causal mask:

```python
masked_scores = [[0.02, -inf], [0.08, 0.07]]
```

Apply softmax:
```python
softmax = [[1.0, 0.0], [0.5025, 0.4975]]
```

---

## Apply Attention to V

Use attention weights to combine V:
```python
Output = attention_weights @ V = [
  [0.2, 0.6],
  [0.24975, 0.5005]
]
```

---

## FeedForward Network (FFN)

A small neural net refines each row:
```python
FFN_output = [
  [0.6, 0.0],
  [0.5005, -0.1493]
]
```

---

## Final Projection to Vocabulary

Use last token output to project to vocab logits:
```python
W_vocab = [
  [0.5, 1.0, -1.0, 0.3, 0.0],
  [-0.5, 0.0, 1.0, -0.2, 0.5]
]
```

Multiply:
```python
logits = [0.3249, 0.5005, -0.6498, 0.1800, -0.0747]
```

Apply softmax → predict token with highest prob: "The"

---

## Python Implementation
Check file PicoGPT.py


---

## Training the Toy Model
Check PicoGPT_training.py

---

## Scaling to ChatGPT

- Larger vocab
- Bigger embeddings
- Longer contexts
- Multi-head attention
- Deep layers
- Training on huge corpora
- Fine-tuning with human feedback

---

## Final Words

Isn't this marvelous? Companies have spent billions creating these models that we can now use daily.

If you enjoyed this article and want to talk more about LLMs, contact:

**Alex Louizos**  
AI Engineer  
alex@maxnmachina.com

