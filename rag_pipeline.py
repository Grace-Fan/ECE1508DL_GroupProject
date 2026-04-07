import faiss
import os
import torch
from sentence_transformers import CrossEncoder
from chunking import embed_chunks, embedding_model
from openai import OpenAI

dimension = embedding_model.get_sentence_embedding_dimension()
print("Embedding dim:", dimension)
index = faiss.IndexFlatIP(dimension)

chunk_store = []

# Add chunks to index
def add_to_index(chunks, title):

    embeddings = embed_chunks(chunks)

    index.add(embeddings)

    for c in chunks:
        chunk_store.append({
            "text": c,
            "title": title
        })
        
# Retrieval
def retrieve(query, top_k=10, score_threshold=0.3):

    query_vec = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, ids = index.search(query_vec, top_k * 3)

    retrieved = []
    for score, i in zip(scores[0], ids[0]):

        if i == -1 or i >= len(chunk_store):
            continue

        if score < score_threshold:
            continue

        retrieved.append({**chunk_store[i], "score": float(score)})

    print(f"  [DEBUG] Retrieved {len(retrieved)} chunks above threshold {score_threshold}")
    return retrieved[:top_k]


# Reranker
# Reranker is loaded lazily after embedding is freed from GPU
reranker = None

def _get_reranker():
    global reranker
    if reranker is None:
        print("Loading reranker on GPU (float16)...")
        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-gemma",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16}
        )
        print("✓ Reranker loaded on GPU")
    return reranker

def free_reranker():
    """Move reranker to CPU and free GPU VRAM for next indexing phase."""
    global reranker
    if reranker is not None:
        reranker.model.to("cpu")
        torch.cuda.empty_cache()
        print("✓ Reranker moved to CPU, GPU VRAM freed")


def rerank(query, docs):

    pairs = [[query, d["text"]] for d in docs]

    scores = _get_reranker().predict(pairs, batch_size=1)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in ranked]



# LLM generator
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=30.0,
    max_retries=2
)


def generate_answer(query, context):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using ONLY the provided context. Be concise. Do not explain."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Answer:
"""
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# RAG pipeline
def rag_pipeline(query):

    docs = retrieve(query, top_k=10)

    docs = rerank(query, docs)

    context = "\n\n".join([
    f"Title: {d['title']}\nContent: {d['text']}"
    for d in docs[:5]
])

    # DEBUG PRINTS (add here)
    print("\n[DEBUG] Query:", query)
    print("[DEBUG] Index size:", index.ntotal)
    if docs:
        print("[DEBUG] Retrieved IDs:", [i for i, _ in enumerate(docs)])
        print("[DEBUG] Top doc titles:", [d["title"] for d in docs[:5]])
    print("[DEBUG] Context:\n", context)
    
    answer = generate_answer(query, context)
    

    return answer, docs