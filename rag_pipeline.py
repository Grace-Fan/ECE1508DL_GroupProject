import faiss
from transformers import pipeline
from sentence_transformers import CrossEncoder
from chunking import embed_chunks, embedding_model

dimension = embedding_model.get_sentence_embedding_dimension()

index = faiss.IndexFlatIP(dimension)

chunk_store = []


# ------------------------------
# Add chunks to index
# ------------------------------
def add_to_index(chunks, title):

    embeddings = embed_chunks(chunks)

    index.add(embeddings)

    for c in chunks:
        chunk_store.append({
            "text": c,
            "title": title
        })


# ------------------------------
# Retrieval
# ------------------------------
def retrieve(query, top_k=10):

    query_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, ids = index.search(query_vec, top_k)

    retrieved = []

    for i in ids[0]:

        if i == -1:
            continue

        if i >= len(chunk_store):
            continue

        retrieved.append(chunk_store[i])

    return retrieved


# ------------------------------
# Reranker
# ------------------------------
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-gemma"
)


def rerank(query, docs):

    pairs = [[query, d["text"]] for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in ranked]


# ------------------------------
# LLM generator
# ------------------------------
# FIXME: change to API for deepseek
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)


def generate_answer(query, context):

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    output = generator(
        prompt,
        max_new_tokens=100
    )

    generated = output[0]["generated_text"]

    answer = generated.split("Answer:")[-1].strip()

    return answer


# ------------------------------
# RAG pipeline
# ------------------------------
def rag_pipeline(query):

    docs = retrieve(query, top_k=10)

    docs = rerank(query, docs)

    context = "\n".join([d["text"] for d in docs[:3]])

    answer = generate_answer(query, context)

    return answer, docs