# テキストの分割を上手く行うことに成功した
# OpenAIのAPIが使えていないっていうエラーが出たが、コード変更せず何回か実行してたら通った
# 無事実行完了
# pip install "ragas[openai]==0.1.0”エラー出たらこれ入れて！！
import time, requests, wandb
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, context_recall, faithfulness
)
# openAIのモデルも評価できるようにしてみる
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def rag_generate_openai(question: str, contexts: list[str], model_name: str) -> str:
    ctx_block = "\n\n".join(f"Document {i+1}:\n{c}" for i, c in enumerate(contexts))
    prompt = (
        f"Please answer the following question based on the documents below.\n\n"
        f"{ctx_block}\n\nQuestion: {question}\n\nAnswer:"
    )

    response = client.chat.completions.create(model=model_name,  # "gpt-3.5-turbo" でも可
    messages=[{"role": "user", "content": prompt}],
    temperature=0)
    return response.choices[0].message.content.strip()
# ---------------- 設定 ----------------
OLLAMA_API_URL = "http://ollama:11434/api/generate"
# MODEL_NAME     = "gemma3:12b"
# MODEL_NAME     = "llama3.3:70b"
MODEL_NAME     = "llama3:8b"
# MODEL_NAME     = "gemma3:27b"
# MODEL_NAME = "gpt-4o"  # OpenAI モデルを使いたい場合



N_SAMPLES      = 10
TOP_K          = 5
PASSAGE_LEN    = 1
# --------------------------------------
# ① データ読み込み
ds_eval   = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{N_SAMPLES}]")
ds_corpus = load_dataset("hotpot_qa", "fullwiki", split="train[:3000]+validation")
# ds_corpus = load_dataset("hotpot_qa", "fullwiki", split="train[:30]")
# EVAL_NAME をモデルとパラメータから動的に生成
EVAL_NAME = f"{MODEL_NAME.replace(':', '_')}_top{TOP_K}_n{N_SAMPLES}_len{PASSAGE_LEN}_corpus{len(ds_corpus)}"

wandb.init(
    # mode="disabled",  # ← ログを取りたくないときは "disabled"
    project="rag-eval-hotpotQA", 
    name=EVAL_NAME, config={
    "model": MODEL_NAME,
    "samples": N_SAMPLES,
    "top_k": TOP_K,
})

sample_table = wandb.Table(columns=["question", "contexts", "answer", "reference"])


# ② コーパス作成（文を 4 本ずつまとめて 1 パッセージ）
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

flat_docs = []
for ex in ds_corpus:
    titles = ex["context"]["title"]
    passages = ex["context"]["sentences"]

    for title, sents in zip(titles, passages):
        for chunk in chunked(sents, PASSAGE_LEN):
            flat_docs.append(f"{title}: " + " ".join(chunk))

# print("総パッセージ数:", len(flat_docs))
print("作成されたパッセージ数:", len(flat_docs))
# print("サンプル:")
# for i in range(10):
#     print(f"[{i}] {flat_docs[i]}")

# ③ 埋め込み + 最近傍インデックス
embedder = SentenceTransformer("all-mpnet-base-v2",device="cuda")
doc_vecs = embedder.encode(flat_docs,batch_size=128, show_progress_bar=True, num_workers=4)
index    = NearestNeighbors(n_neighbors=TOP_K, metric="cosine").fit(doc_vecs)

def retrieve(question: str):
    q_vec = embedder.encode([question])
    _, idxs = index.kneighbors(q_vec)
    return [flat_docs[i] for i in idxs[0]]

def rag_generate(question: str, contexts: list[str]) -> str:
    ctx_block = "\n\n".join(f"【文書{i+1}】\n{c}" for i, c in enumerate(contexts))
    # prompt = f"以下の文書を参考に質問に答えてください。\n\n{ctx_block}\n\n質問: {question}\n\n答え:"
    prompt = f"Please answer the following question based on the documents below.\n\n{ctx_block}\n\nQuestion: {question}\n\nAnswer:"
    r = requests.post(OLLAMA_API_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
    r.raise_for_status()
    return r.json()["response"].strip()
    # return r.json().response.strip()

# ④ 実行ループ
records = []
t0 = time.time()
for i, ex in enumerate(ds_eval):
    q   = ex["question"]
    gt  = ex["answer"]
    ctx = retrieve(q)
    # print(ctx)
    # print(gt)
    if MODEL_NAME.lower() in ["gpt-3.5-turbo", "gpt-4o"]: #ここは使うモデルに応じて追加する必要がある
        # print("openAI!!")
        ans = rag_generate_openai(q, ctx, MODEL_NAME)
    else:
        ans = rag_generate(q, ctx)  # Ollama用
    # ans = rag_generate(q, ctx)
    sample_table.add_data(q, ctx, ans, gt)
    records.append({
        "question":      q,
        "contexts":      ctx,
        "answer":        ans,
        "ground_truth":  gt[0],
        "reference":     gt
    })
    print(f"[{i+1}/{N_SAMPLES}] done")

print("処理時間:", round(time.time() - t0, 1), "秒")

# ⑤ RAGAS 評価
rag_ds  = Dataset.from_list(records)
# metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
metrics = [answer_relevancy, faithfulness, context_precision]
results = evaluate(rag_ds, metrics=metrics)

print(results)
# 評価結果を辞書として取得
print("\n===== RAGAS scores =====")

scores_dict = {}
for name in results:               # name は 'answer_relevancy' などの文字列
    val = results[name]            # スコアを取得
    print(f"{name:20s}: {val:.3f}")
    scores_dict[name] = val        # wandb 用に保存


wandb.log({**scores_dict, "samples": sample_table})
wandb.finish()




# # hotpotQAデータセットが動いた(wandbとRAGAS対応していない)
# # wandbを対応させる
# import time, requests, wandb
# from datasets import load_dataset, Dataset
# from sentence_transformers import SentenceTransformer
# from sklearn.neighbors import NearestNeighbors
# from ragas import evaluate
# from ragas.metrics import (
#     answer_relevancy, context_precision, context_recall, faithfulness
# )
# # ---------------- 設定 ----------------
# OLLAMA_API_URL = "http://ollama:11434/api/generate"
# # MODEL_NAME = "gemma3:12b"
# MODEL_NAME = "llama3:8b"
# EVAL_NAME      = "ragas-gemma3-amnesty"
# N_SAMPLES  = 2          # 評価サンプル数
# TOP_K      = 5          # 検索で返すパッセージ数
# PASSAGE_LEN = 4         # 文をまとめる数
# # --------------------------------------

# wandb.init(project="rag-eval", name=EVAL_NAME, config={ "model": MODEL_NAME,
#         # "dataset": DS_NAME,
#         # "config_name": CONFIG_NAME,
#         "samples": N_SAMPLES,
#         "top_k": TOP_K,})

# sample_table = wandb.Table(columns=["question", "contexts", "answer", "reference"])

# # ① データ読み込み
# ds_eval   = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{N_SAMPLES}]")
# ds_corpus = load_dataset("hotpot_qa", "fullwiki", split="train[:3000]+validation")

# # ② コーパス作成（文を 4 本ずつまとめて 1 パッセージ）
# def chunked(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i+n]

# flat_docs = []
# for ex in ds_corpus:
#     for title, sents, *_ in ex["context"]:
#         for chunk in chunked(sents, PASSAGE_LEN):
#             flat_docs.append(f"{title}: " + " ".join(chunk))

# print("総パッセージ数:", len(flat_docs))   # 例: 150 K 以上

# # ③ 埋め込み + 最近傍インデックス
# embedder = SentenceTransformer("all-mpnet-base-v2")   # → 768 次元
# doc_vecs = embedder.encode(flat_docs, show_progress_bar=True)
# index    = NearestNeighbors(n_neighbors=TOP_K, metric="cosine").fit(doc_vecs)

# def retrieve(question: str):
#     q_vec = embedder.encode([question])
#     _, idxs = index.kneighbors(q_vec)
#     return [flat_docs[i] for i in idxs[0]]

# def rag_generate(question: str, contexts: list[str]) -> str:
#     ctx_block = "\n\n".join(f"【文書{i+1}】\n{c}" for i, c in enumerate(contexts))
#     prompt = f"以下の文書を参考に質問に答えてください。\n\n{ctx_block}\n\n質問: {question}\n\n答え:"
#     r = requests.post(OLLAMA_API_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
#     r.raise_for_status()
#     return r.json()["response"].strip()

# # ④ 実行ループ（例：5 件だけ回してみる）
# records = []
# t0 = time.time()
# for i, ex in enumerate(ds_eval):
#     q   = ex["question"]
#     gt  = ex["answer"]
#     ctx = retrieve(q)
#     ans = rag_generate(q, ctx)
#     sample_table.add_data(q, ctx, ans, gt[0])
#     records.append({
#         "question":      q,
#         "contexts":      ctx,
#         "answer":        ans,
#         "ground_truths": gt,
#         "reference":     gt[0]  # 🔥 RAGAS workaround
#     })
#     print(f"[{i+1}/{N_SAMPLES}] done")

# print("処理時間:", round(time.time()-t0, 1), "秒")

# rag_ds  = Dataset.from_list(records)
# metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
# # レコードの結果とメトリックスから結果を計算
# results = evaluate(rag_ds, metrics=metrics)

# elapsed = time.time() - t0


# print("\n===== RAGAS scores =====")
# scores_dict = {}

# # for score in results.scores:  # ← List[dict]
# #     name = score["name"]       # スコア名（例: "faithfulness"）
# #     val  = float(score["score"])  # スコア値
# #     print(f"{name:20s}: {val:.3f}")
# #     scores_dict[name] = val

# scores_dict = {}
# for name in results:               # name は 'answer_relevancy' などの文字列
#     val = results[name]            # スコアを取得
#     print(f"{name:20s}: {val:.3f}")
#     scores_dict[name] = val        # wandb 用に保存

# wandb.log({**scores_dict, "samples": sample_table})
# wandb.finish()
# # `records` をそのまま RAGAS に渡して評価しても良い
