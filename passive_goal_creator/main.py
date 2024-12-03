from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"


class PassiveGoalCreator:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> Goal:
        logging.info(f"PassiveGoalCreator: Running with query: {query}")
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件:\n"
            "1. エンティティを抽出する。(提示されたユースケースの名詞、動詞に着目)\n"
            "2. 洗い出したエンティティを[リソース]と[イベント]に分類する。イベントに分類する基準は属性に”日時・日付（イベントが実行された日時・日付）”を持つものである。\n"
            "3. イベントエンティティには1つの日時属性しかもたないようにする。\n"
            "4. リソースに隠されたイベントを抽出する。（リソースに更新日時をもちたい場合にはイベントが隠されている可能性がある）\n"
            "  例）社員情報（リソース）の更新日時がある場合には、社員異動（イベントエンティティ）を抽出する。\n"
            "5. エンティティ間の依存度が強すぎる場合には、交差エンティティ（関連エンティティ）を導入する。（カーディナリティが多対多の関係を持つような場合に導入する）\n"
            "ユーザーの入力: {query}"
        )

        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})


def main():
    import argparse

    from settings import Settings

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="PassiveGoalCreatorを利用して目標を生成します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        model=settings.openai_smart_model, temperature=settings.temperature
    )
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)

    print(f"{result.text}")


if __name__ == "__main__":
    main()
