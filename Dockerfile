FROM python:3.10-slim

WORKDIR /app

# git をインストール
RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --no-cache-dir \
        ragas[openai]==0.1.0 \
        sentence-transformers \
        faiss-cpu \
        datasets \
        requests \
        wandb

COPY gsm8k_eval.py .

# 出力を画面 & ファイルに保存（eval_log.txt）
# CMD ["bash", "-c", "python gsm8k_eval3.py | tee eval_log.txt"]
# CMD ["python", "gsm8k_eval3.py"]
# CMD ["python", "math_500_eval2.py"]
# CMD ["/bin/sh"] 
# CMD ["/bin/bash"]
