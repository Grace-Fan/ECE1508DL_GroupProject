import torch
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
nltk.download("punkt")
nltk.download("punkt_tab")

# Embedding model loaded in float16 to reduce VRAM usage
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"torch_dtype": torch.float16}
)
embedding_model.max_seq_length = 256

def free_embedding_model():
    """Move embedding model to CPU and free GPU VRAM."""
    embedding_model.to("cpu")
    torch.cuda.empty_cache()
    print("✓ Embedding model moved to CPU, GPU VRAM freed")

def reload_embedding_model():
    """Move embedding model back to GPU for indexing."""
    embedding_model.to("cuda")
    print("✓ Embedding model moved back to GPU")


# 1. Fixed-length chunking
def fixed_chunk(text, chunk_size, overlap=None):

    words = text.split()
    chunks = []

    # Default overlap is 20% of chunk_size; always keep stride >= 1
    if overlap is None:
        overlap = chunk_size // 5
    stride = max(1, chunk_size - overlap)

    start = 0
    while start < len(words):

        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += stride

    return chunks

# 2. Sentence-based chunking

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


# 3. Semantic chunking
def semantic_chunk(text, threshold=0.60, min_chunk_words=30):

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return sentences

    sent_embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    chunks = []
    current_chunk = [sentences[0]]
    anchor_emb = sent_embeddings[0]

    for i in range(1, len(sentences)):

        # Compare against the chunk's anchor (first sentence) not just the previous
        sim = np.dot(anchor_emb, sent_embeddings[i])

        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            anchor_emb = sent_embeddings[i]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Merge chunks that are too short into the previous chunk
    merged = []
    for chunk in chunks:
        if merged and len(merged[-1].split()) < min_chunk_words:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged

# Embedding helper
def embed_chunks(chunks):

    embeddings = embedding_model.encode(
        chunks,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings
