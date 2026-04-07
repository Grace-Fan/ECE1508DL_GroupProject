import gc
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import torch
from datasets import load_dataset
from chunking import fixed_chunk, sentence_chunk, semantic_chunk, free_embedding_model, reload_embedding_model
from rag_pipeline import add_to_index, rag_pipeline, index, chunk_store, client, free_reranker

# Experiment matrix
EXPERIMENTS = [
    {"strategy": "fixed",    "chunk_size": 50},
    {"strategy": "fixed",    "chunk_size": 100},
    {"strategy": "fixed",    "chunk_size": 200},
    {"strategy": "sentence", "chunk_size": 50},
    {"strategy": "sentence", "chunk_size": 100},
    {"strategy": "sentence", "chunk_size": 200},
    {"strategy": "semantic", "chunk_size": None},
]

TRAIN_SIZE = 1000
DEV_SIZE   = 500
EVAL_SIZE  = 50

# Load dataset (once, before all experiments)
print("\n" + "="*80)
print("STEP 1: Loading dataset...")
print("="*80)

train_data = load_dataset("trivia_qa", "rc", split=f"train[:{TRAIN_SIZE}]")
dev_data   = load_dataset("trivia_qa", "rc", split=f"validation[:{DEV_SIZE}]")

train_data = list(train_data)
dev_data   = list(dev_data)
print(f"✓ Loaded {len(train_data)} train entries and {len(dev_data)} dev entries")

# Print one sample of the dataset
print("\n" + "="*80)
print("SAMPLE DATASET ENTRY:")
print("="*80)
sample = train_data[0]
sample_file = "sample_entry.json"
with open(sample_file, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, default=str)
print(f"✓ Sample entry saved to: {sample_file}")

all_entries  = train_data + dev_data
eval_entries = dev_data[:EVAL_SIZE]

# Print all questions and gold answers (once, same across all experiments)
print("\n" + "="*80)
print(f"STEP 4: All Questions and Gold Answers (dev set, first {EVAL_SIZE})")
print("="*80)
for i, entry in enumerate(eval_entries):
    q = entry.get("question", "N/A")
    a = entry.get("answer", {}).get("value", "N/A")
    print(f"\n[{i+1}] Question   : {q}")
    print(f"     Gold Answer: {a}")

# Load existing results (resume support)
results_file = "rag_results.csv"
if os.path.exists(results_file):
    df_existing = pd.read_csv(results_file)
    results = df_existing.to_dict("records")
    print(f"\n✓ Resuming: {len(results)} experiments already in {results_file}")
else:
    results = []

# Evaluation helpers
squad_metric = evaluate.load("squad")

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def compute_f1(pred, gold):
    result = squad_metric.compute(
        predictions=[{"id": "1", "prediction_text": pred}],
        references=[{"id": "1", "answers": {"text": [gold], "answer_start": [0]}}]
    )
    return result["f1"]

def llm_judge(question, prediction, gold):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "Evaluate if the predicted answer matches the ground truth. Reply only with 'Score: 1' or 'Score: 0'."
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nGround Truth Answer:\n{gold}\n\nPredicted Answer:\n{prediction}"
            }
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    return 1 if "Score: 1" in output else 0

# Main experiment loop
for exp_idx, exp in enumerate(EXPERIMENTS):

    chunk_strategy = exp["strategy"]
    chunk_size     = exp["chunk_size"]
    chunk_size_label = chunk_size if chunk_size is not None else 0

    # Skip if already completed in a previous run
    already_done = any(
        r["strategy"] == chunk_strategy and r["chunk_size"] == chunk_size_label
        for r in results
    )
    if already_done:
        print(f"\n⏭  Skipping {chunk_strategy} chunk_size={chunk_size_label} (already in results)")
        continue

    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_idx+1}/{len(EXPERIMENTS)}: strategy={chunk_strategy}, chunk_size={chunk_size_label}")
    print("="*80)

    # --- GPU swap: bring embedding model back for indexing ---
    if exp_idx > 0:
        reload_embedding_model()

    # 4. Build Index
    print("\n" + "="*80)
    print(f"STEP 3: Building Index  [strategy={chunk_strategy}, chunk_size={chunk_size_label}]")
    print("="*80)
    index.reset()
    chunk_store.clear()
    torch.cuda.empty_cache()
    print("✓ Index and chunk store cleared")

    total_chunks   = 0
    total_passages = 0

    for entry_idx, entry in enumerate(all_entries):
        results_list = entry.get("search_results", {}).get("search_context", [])
        titles_list  = entry.get("search_results", {}).get("title", [])

        if entry_idx < 3:
            print(f"  [DEBUG] Entry {entry_idx}: {len(titles_list)} titles, {len(results_list)} passages")

        t0 = time.time()

        for title, section in zip(titles_list, results_list):
            passage_text = f"{title}. {section}"
            if not passage_text.strip():
                continue

            total_passages += 1

            if chunk_strategy == "fixed":
                chunks = fixed_chunk(passage_text, chunk_size)
            elif chunk_strategy == "sentence":
                chunks = sentence_chunk(passage_text, chunk_size)
            else:
                chunks = semantic_chunk(passage_text)

            total_chunks += len(chunks)
            add_to_index(chunks, title=title or "unknown")

        if (entry_idx + 1) % 50 == 0:
            print(f"  → Processed {entry_idx + 1}/{len(all_entries)} entries | "
                  f"Passages: {total_passages} | Chunks: {total_chunks} | "
                  f"Time: {time.time()-t0:.2f}s")

    print(f"\n✓ Chunking complete      : {total_chunks} total chunks from {total_passages} passages")
    print(f"✓ Database build complete: index size = {index.ntotal} vectors")

    # --- GPU swap: free embedding model, reranker will load on GPU ---
    free_embedding_model()


    # 5. Evaluation
    print("\n" + "="*80)
    print(f"STEP 5: Running Evaluation  [strategy={chunk_strategy}, chunk_size={chunk_size_label}]")
    print("="*80)

    recalls, ems, f1s, llm_scores = [], [], [], []
    raw_answers = []

    for i, entry in enumerate(eval_entries):

        query = entry.get("question")
        if not query:
            print("[WARNING] Empty query, skipping entry")
            continue

        gold_answer = entry["answer"]["value"]
        gold_titles = entry.get("search_results", {}).get("title", [])

        print(f"\n[{i+1}/{EVAL_SIZE}] QUESTION ASKED : {query}")

        answer, docs = rag_pipeline(query)
        print(f"      ✓ Retrieved {len(docs)} chunks")

        print(f"\n      GOLD ANSWER  : {gold_answer}")
        print(f"      MODEL ANSWER : {answer}")

        score = llm_judge(query, answer, gold_answer)
        print(f"      ✓ LLM Judge Score: {score}")

        hit = any(d["title"] in gold_titles for d in docs)
        recalls.append(hit)
        ems.append(exact_match(answer, gold_answer))
        f1s.append(compute_f1(answer, gold_answer))
        llm_scores.append(score)
        raw_answers.append({
            "question":     query,
            "gold_answer":  gold_answer,
            "model_answer": answer,
            "llm_judge":    score,
            "em":           ems[-1],
            "f1":           f1s[-1],
            "recall_hit":   hit,
            "top_doc_titles": " | ".join([d["title"] for d in docs[:3]]),
            "top_chunk":    docs[0]["text"][:300] if docs else ""
        })

        print(f"      Recall Hit: {hit} | EM: {ems[-1]} | F1: {f1s[-1]:.4f}")
        print("-"*80)

    # 6: Save results incrementally
    results.append({
        "strategy":   chunk_strategy,
        "chunk_size": chunk_size_label,
        "recall@10":  sum(recalls)    / len(recalls),
        "EM":         sum(ems)        / len(ems),
        "F1":         sum(f1s)        / len(f1s),
        "LLM_eval":   sum(llm_scores) / len(llm_scores)
    })

    # Save raw per-question answers for potential re-judging later
    raw_file = f"raw_answers_{chunk_strategy}_{chunk_size_label}.csv"
    pd.DataFrame(raw_answers).to_csv(raw_file, index=False)
    print(f"✓ Raw answers saved to {raw_file}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")
    print(df_results.tail())

    # --- GPU swap: free reranker so next iteration can embed on GPU ---
    if exp_idx < len(EXPERIMENTS) - 1:
        free_reranker()
        gc.collect()
        torch.cuda.empty_cache()

# 7: Plot all results
print("\n" + "="*80)
print("STEP 7: Generating Plots...")
print("="*80)

df_results = pd.read_csv(results_file)
os.makedirs("plots", exist_ok=True)
plot_metrics = ["recall@10", "EM", "F1", "LLM_eval"]

for plot_metric in plot_metrics:

    plt.figure()

    for strategy in df_results["strategy"].unique():
        subset = df_results[df_results["strategy"] == strategy].sort_values("chunk_size")
        plt.plot(subset["chunk_size"], subset[plot_metric], marker="o", label=strategy)

    plt.xlabel("Chunk Size (0 = semantic)")
    plt.ylabel(plot_metric)
    plt.title(f"{plot_metric} vs Chunk Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"plots/{plot_metric}_vs_chunk_size.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved plot: {filename}")
    plt.close()

print("\n" + "="*80)
print("✓ ALL EXPERIMENTS COMPLETE")
print("="*80)
print(df_results.to_string(index=False))
