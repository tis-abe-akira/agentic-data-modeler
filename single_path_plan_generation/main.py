import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from passive_goal_creator.main import Goal, PassiveGoalCreator
from prompt_optimizer.main import OptimizedGoal, PromptOptimizer
from pydantic import BaseModel, Field
from response_optimizer.main import ResponseOptimizer
import logging
from settings import Settings
settings = Settings()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3~5個に分解されたタスク",
    )


class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        logging.info(f"QueryDecomposer: Running with query: {query}")
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            "   - Amazon Bedrock KnowledgeBaseを利用して、イミュータブルデータモデルに関する知識を取得する。\n"
            "   - インターネットを利用して、目標を達成するための一般公開情報の調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. タスクは日本語で出力すること。\n"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=settings.KNOWLEDGE_BASE_ID,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
        )
        tool = create_retriever_tool(
            retriever,
            "BedrockKnowledgeBasesRetriever",
            "With this tool, you can search the Amazon Bedrock KnowledgeBase to gain insights into Immutable Data Modeling.",
        )

        self.tools = [
            tool,
            TavilySearchResults(max_results=3)
        ]

    def run(self, task: str) -> str:
        logging.info(f"---- TaskExecutor: Running with task: {task}")
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "次のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task}\n\n"
                            "要件:\n"
                            "1. 必要に応じて提供されたツールを使用してください。\n"
                            "2. 実行は徹底的かつ包括的に行ってください。\n"
                            "3. 可能な限り具体的な事実やデータを提供してください。\n"
                            "4. 発見した内容を明確に要約してください。\n"
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content


class ResultAggregator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, optimized_goal: str, response_definition: str, results: list[str]) -> str:
        logging.info(f"ResultAggregator: Running with query: {query}")
        logging.info(f"ResultAggregator: Running with optimized_goal: {optimized_goal}")
        logging.info(f"ResultAggregator: Running with response_definition: {response_definition}")
        logging.info(f"ResultAggregator: Running with results: {results}")
        prompt = ChatPromptTemplate.from_template(
            "ユーザー入力:\n{query}\n\n"
            "与えられた目標:\n{optimized_goal}\n\n"
            "調査結果:\n{results}\n\n"
            "ユーザー入力と与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "Mermeid形式のER図のコードは、省略せずにそのまま出力してください。\n"
            "Mermeid形式のER図のコードに限り、英語で出力してください。\n"
            "{response_definition}"
        )
        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(results)
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "optimized_goal": optimized_goal,
                "results": results_str,
                "response_definition": response_definition,
            }
        )


class SinglePathPlanGeneration:
    def __init__(self, llm: ChatOpenAI):
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.query_decomposer = QueryDecomposer(llm=llm)
        self.task_executor = TaskExecutor(llm=llm)
        self.result_aggregator = ResultAggregator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(SinglePathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"},
        )
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        # プロンプト最適化
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        # レスポンス最適化
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(
            query=state.optimized_goal
        )
        logging.info("***** _decompose_query: ")
        logging.info(f"Decomposed tasks: {decomposed_tasks}")
        # 分解されたタスクが最大許容長さを超えないようにする TODO:エラーが出たのでCopilotに相談して修正した。
        # if len(decomposed_tasks.values) > 5:
        #     decomposed_tasks.values = decomposed_tasks.values[:5]
        return {"tasks": decomposed_tasks.values}

    def _execute_task(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(
        self, state: SinglePathPlanGenerationState
    ) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.query,
            optimized_goal=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        initial_state = SinglePathPlanGenerationState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 2000})
        return final_state.get("final_output", "Failed to generate a final response.")


def main():
    import argparse
    import re
    import requests
    import tempfile
    import os

    parser = argparse.ArgumentParser(
        description="SinglePathPlanGenerationを使用してタスクを実行します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    agent = SinglePathPlanGeneration(llm=llm)
    result = agent.run(args.task)
    print(result)

    # Mermaidコードブロックを検出
    mermaid_match = re.search(r'```mermaid\n(.*?)\n```', result, re.DOTALL)
    if mermaid_match:
        mermaid_code = mermaid_match.group(1)
        # Kroki APIを使用して画像を取得
        response = requests.post(
            'https://kroki.io/mermaid/png',
            data=mermaid_code.encode('utf-8'),
            headers={'Content-Type': 'text/plain'}
        )
        if response.status_code == 200:
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(response.content)
                image_filepath = tmp_file.name
            # 画像を表示
            os.system(f'open "{image_filepath}"')
        else:
            print("Mermaid図の生成に失敗しました。")

if __name__ == "__main__":
    main()
