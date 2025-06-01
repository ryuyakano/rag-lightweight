# openAIのapiを使用した実験を行う

import openai
import requests
import json
import time
import re
from datasets import load_dataset

import wandb  


OLLAMA_API_URL = "http://ollama:11434/api/generate"
# MODEL_NAME = "llama3:8b"
# MODEL_NAME = "llama3.3:70b"
MODEL_NAME = "gemma3:27b"
# MODEL_NAME = "gemma3:12b"

HEADERS = {"Content-Type": "application/json"}

# 最初の10問でテスト
dataset = load_dataset("HuggingFaceH4/MATH-500", split="test[:50]")

# LLMに判断させる用の関数を追加
def ask_judgment(question, expected, reply):
    judge_prompt = (
        f"[問題]\n{question}\n\n"
        f"[期待される正解]\n{expected}\n\n"
        f"[モデルの出力]\n{reply}\n\n"
        "このモデルの出力が期待される正解と一致するかどうかを「正解」または「不正解」で答えてください。"
        "その理由も簡潔に説明してください。"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": judge_prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, headers=HEADERS, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "").strip()
    else:
        return f"判定失敗: {response.status_code}"

# wandb 初期化
wandb.init(project="math-llm-eval", name=f"eval-{MODEL_NAME}", config={
    "model": MODEL_NAME,
    "dataset": "MATH-500",
    "num_samples": 50
})


def ask_judgment_openai(question, expected, reply):
    judge_prompt = (
        f"[問題]\n{question}\n\n"
        f"[期待される正解]\n{expected}\n\n"
        f"[モデルの出力]\n{reply}\n\n"
        "このモデルの出力が期待される正解と一致するかどうかを「正解」または「不正解」で答えてください。"
        "その理由も簡潔に説明してください。"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは厳格な数式採点者です。正しい場合のみ正解と判断してください。"},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        return f"判定失敗（OpenAI APIエラー）: {e}"

# correct = 0
regex_correct = 0
llm_correct = 0
total_time = 0
max_time = 0
max_time_problem = ""

# --- 評価ループ ---
for i, example in enumerate(dataset, 1):
    question = example['problem'].strip()
    expected_raw = example['solution']
    expected_match = re.search(r"\\boxed\{(.+?)\}", expected_raw)
    expected_answer = expected_match.group(1).strip() if expected_match else None

    print(f"[{i}] 問題: {question}")
    print(f"     正解: {expected_answer}")

    prompt = (
        f"{question}\n"
        "Please solve the problem step-by-step. "
        "Then, write your final answer using LaTeX box format like \\boxed{{...}}."
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, headers=HEADERS, data=json.dumps(payload))
    except Exception as e:
        print(f"通信エラー: {e}")
        continue
    elapsed_time = time.time() - start_time
    total_time += elapsed_time

    if elapsed_time > max_time:
        max_time = elapsed_time
        max_time_problem = question

    if response.status_code == 200:
        result = response.json()
        reply = result.get("response", "").strip()
        print("  モデルの回答:", reply)

        match = re.search(r"\\boxed\{(.+?)\}", reply)
        predicted_answer = match.group(1).strip() if match else None

        # --- 評価 ---
        is_regex_correct = False
        is_llm_correct = False

        if predicted_answer == expected_answer:
            print("正規表現一致で正解")
            is_regex_correct = True
            regex_correct += 1
        else:
            if predicted_answer:
                print(f"正規表現一致せず（抽出: {predicted_answer}）")
            else:
                print("フォーマット不一致（\\boxed{{...}} が見つかりません）")

            # LLM による柔軟評価を実施
            # judgment = ask_judgment(question, expected_answer, reply)
            judgment = ask_judgment_openai(question, expected_answer, reply)
            print("LLMによる判定:", judgment)

            if "正解" in judgment and "不正解" not in judgment:
                print("LLM評価で正解（柔軟判定）")
                is_llm_correct = True
                llm_correct += 1
            else:
                print("LLM評価では不正解")

        print()
    else:
        print(f"APIエラー: {response.status_code}\n{response.text}")

    time.sleep(1)

# 結果出力
total = len(dataset)
regex_accuracy = regex_correct / total * 100
combined_accuracy = (regex_correct + llm_correct) / total * 100
average_time = total_time / total

print("評価結果")
print(f"モデル名: {MODEL_NAME}")
print(f"正規表現一致の正解数: {regex_correct}/{total}（{regex_accuracy:.2f}%）")
print(f"LLMによる追加正解数: {llm_correct}（除外済み）")
print(f"合計正解数（重複なし）: {regex_correct + llm_correct}/{total}（{combined_accuracy:.2f}%）")
print(f"平均応答時間: {average_time:.2f} 秒")
print(f"最も時間がかかった問題:\n{max_time_problem}")
print(f"所要時間: {max_time:.2f} 秒")

wandb.log({
    "total_samples": total,
    "regex_correct_total": regex_correct,
    "llm_correct_total": llm_correct,
    "regex_accuracy": regex_accuracy,
    "combined_accuracy": combined_accuracy,
    "average_response_time": average_time,
    "max_time_problem": max_time_problem,
    "max_time": max_time
})

wandb.finish()