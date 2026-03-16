import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

# Embedding model
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B"
)

# ------------------------------
# 1. Fixed-length chunking
# ------------------------------
def fixed_chunk(text, chunk_size, overlap=50):

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):

        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ------------------------------
# 2. Sentence-based chunking
# ------------------------------
def sentence_chunk(text, chunk_size):

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    length = 0

    for s in sentences:

        tokens = len(s.split())

        if length + tokens > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            length = 0

        current_chunk.append(s)
        length += tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ------------------------------
# 3. Semantic chunking
# ------------------------------
def semantic_chunk(text, threshold=0.75):

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return sentences

    sent_embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = np.dot(sent_embeddings[i-1], sent_embeddings[i])

        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ------------------------------
# Embedding helper
# ------------------------------
def embed_chunks(chunks):

    embeddings = embedding_model.encode(
        chunks,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings