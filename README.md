# Agentic data modeler

AIエージェントによるデータモデリングツールです。

## 説明はこちら

[TBD]

## ライブラリインストール

前提として `uv` がインストールされていること。

```console
uv add langchain-core==0.3.0 langchain-community==0.3.0 langgraph==0.2.22 langchain-openai==0.2.0 langchain-aws==0.2.0 python-dotenv==1.0.1 numpy==1.26.4 faiss-cpu==1.8.0.post1 pydantic-settings==2.5.2 retry==0.9.2 decorator==4.4.2
```

## 実行

指示文章が長い場合には、ファイル (`task.txt` など） に書いて、そのファイルを引数に指定する。

```console
uv run python -m single_path_plan_generation.main --task "$(cat task.txt)"
```
