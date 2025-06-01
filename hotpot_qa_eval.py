# 実行は成功したが検索がうまくいかないため生成もできていない
import os, time, requests, wandb
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ------------ 設定 ----------------
OLLAMA_API_URL = "http://ollama:11434/api/generate"
MODEL_NAME     = "gemma3:12b"
EVAL_NAME      = "ragas-gemma3-hotpot"
N_SAMPLES      = 3
TOP_K          = 5
# ----------------------------------

# ① HotpotQA fullwiki / validation の先頭 N 件
ds = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{N_SAMPLES}]")

# ② wandb 初期化
wandb.init(project="rag-eval", name=EVAL_NAME, config={
    "model": MODEL_NAME,
    "dataset": "hotpot_qa_fullwiki",
    "samples": N_SAMPLES,
    "top_k": TOP_K,
})
sample_table = wandb.Table(columns=["question", "contexts", "answer", "reference"])

# ③ Retriever 準備
embedder = SentenceTransformer("all-MiniLM-L6-v2")

flat_docs = []
for ex in ds:
    for ctx_item in ex["context"]:
        title   = ctx_item[0]
        sents   = ctx_item[1]        # 2 番目が文リスト
        # 3 番目以降（ctx_item[2:]）は無視で OK
        flat_docs.extend([f"{title}: {sent}" for sent in sents])


doc_vecs = embedder.encode(flat_docs, show_progress_bar=True)
index = NearestNeighbors(n_neighbors=TOP_K, metric="cosine").fit(doc_vecs)

def retrieve(question: str):
    q_vec = embedder.encode([question])
    _, idxs = index.kneighbors(q_vec)
    return [flat_docs[i] for i in idxs[0]]

def rag_generate(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"【文書{i+1}】\n{c}" for i, c in enumerate(contexts)])
    prompt = (f"以下の文書を参考に質問に答えてください。\n\n"
              f"{context_block}\n\n質問: {question}\n\n答え:")
    r = requests.post(OLLAMA_API_URL,
                      json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# ④ RAG 実行
records, t0 = [], time.time()
for i, ex in enumerate(ds):
    q   = ex["question"]
    gt  = [ex["answer"]]
    ctx = retrieve(q)
    ans = rag_generate(q, ctx)

    sample_table.add_data(q, ctx, ans, gt[0])
    records.append({
        "question":      q,
        "contexts":      ctx,
        "answer":        ans,
        "ground_truths": gt,
        "reference":     gt[0],   # RAGAS が一部 reference を期待するため
    })
    print(f"[{i+1}/{N_SAMPLES}] done")

# ⑤ RAGAS 評価
rag_ds  = Dataset.from_list(records)
metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
results = evaluate(rag_ds, metrics=metrics)

# スコア表示 & wandb 保存
print("\n===== RAGAS scores =====")
scores_dict = {name: float(results[name]) for name in results}
for k, v in scores_dict.items():
    print(f"{k:20s}: {v:.3f}")

wandb.log({**scores_dict, "samples": sample_table})
wandb.finish()
