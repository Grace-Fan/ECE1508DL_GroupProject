import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import evaluate

from datasets import load_dataset
from chunking import fixed_chunk, sentence_chunk, semantic_chunk
from rag_pipeline import add_to_index, rag_pipeline, index, chunk_store, generator

# ------------------------------
# Load dataset
# ------------------------------
dataset = load_dataset("kilt_tasks", "triviaqa_support_only")

train_data = dataset["train"]
dev_data = dataset["validation"]

train_data = [json.loads(x) if isinstance(x, str) else x for x in train_data]
dev_data = [json.loads(x) if isinstance(x, str) else x for x in dev_data]

# ------------------------------
# Experiment settings
# ------------------------------
chunk_strategy = "fixed"
chunk_size = 200

print(f"\nEvaluating {chunk_strategy} chunking | size {chunk_size}")

# ------------------------------
# Build index
# ------------------------------
index.reset()
chunk_store.clear()

for entry in train_data[:2000]:

    if not entry["output"]:
        continue

    prov_list = entry["output"][0].get("provenance", [])

    for prov in prov_list:

        passage_text = prov.get("text")

        if not passage_text:
            continue

        if chunk_strategy == "fixed":
            chunks = fixed_chunk(passage_text, chunk_size)

        elif chunk_strategy == "sentence":
            chunks = sentence_chunk(passage_text, chunk_size)

        else:
            chunks = semantic_chunk(passage_text)

        add_to_index(
            chunks,
            title=prov.get("title", "unknown")
        )

# ------------------------------
# LLM Judge
# ------------------------------
def llm_judge(question, prediction, gold):

    prompt = f"""
You are evaluating a question answering system.

Question:
{question}

Ground Truth Answer:
{gold}

Predicted Answer:
{prediction}

Is the predicted answer correct?

Respond ONLY with:
Score: 1
or
Score: 0
"""

    result = generator(prompt, max_new_tokens=50)

    output = result[0]["generated_text"]

    if "Score: 1" in output:
        return 1
    else:
        return 0


# ------------------------------
# Evaluation
# ------------------------------
metric = evaluate.load("squad")

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def compute_f1(pred, gold):

    result = metric.compute(
        predictions=[{"id":"1","prediction_text":pred}],
        references=[{"id":"1","answers":{"text":[gold],"answer_start":[0]}}]
    )

    return result["f1"]

recalls, ems, f1s, llm_scores = [], [], [], []

for entry in dev_data[:50]:

    query = entry.get("input")
    gold_answer = entry["output"][0]["answer"]
    gold_title = entry["output"][0]["provenance"][0]["title"]

    answer, docs = rag_pipeline(query)

    print("\n" + "="*80)
    print("QUESTION:")
    print(query)

    print("\nTOP RETRIEVED CHUNKS:")

    for i, d in enumerate(docs[:5]):

        print(f"\nRank {i+1}")
        print("Title:", d["title"])
        print("Text:", d["text"][:300])
        print("-"*40)

    print("\nGENERATED ANSWER:")
    print(answer)

    print("\nGOLD ANSWER:")
    print(gold_answer)

    hit = any(d["title"] == gold_title for d in docs)

    recalls.append(hit)

    ems.append(exact_match(answer, gold_answer))

    f1s.append(compute_f1(answer, gold_answer))

    score = llm_judge(query, answer, gold_answer)
    llm_scores.append(score)


# ------------------------------
# Save results
# ------------------------------
results_file = "rag_results.csv"

if os.path.exists(results_file):

    df_existing = pd.read_csv(results_file)
    results = df_existing.to_dict("records")

else:

    results = []

results.append({

    "strategy": chunk_strategy,
    "chunk_size": chunk_size,
    "recall@10": sum(recalls)/len(recalls),
    "EM": sum(ems)/len(ems),
    "F1": sum(f1s)/len(f1s),
    "LLM_eval": sum(llm_scores)/len(llm_scores)

})

df_results = pd.DataFrame(results)
df_results.to_csv(results_file, index=False)

print("\nSaved experiment results:")
print(df_results.tail())

# ------------------------------
# Plot results
# ------------------------------
os.makedirs("plots", exist_ok=True)

metrics = ["recall@10", "EM", "F1", "LLM_eval"]

for metric in metrics:

    plt.figure()

    for strategy in df_results["strategy"].unique():

        subset = df_results[df_results["strategy"] == strategy]
        subset = subset.sort_values("chunk_size")

        plt.plot(
            subset["chunk_size"],
            subset[metric],
            marker="o",
            label=strategy
        )

    plt.xlabel("Chunk Size")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Chunk Size")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"plots/{metric}_vs_chunk_size.png"

    plt.savefig(filename, dpi=300)

    print(f"Saved plot: {filename}")

    plt.show()
