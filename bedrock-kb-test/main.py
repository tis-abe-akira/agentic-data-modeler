"""
Amazon Bedrock KnowledgeBases retriever.
動作確認用のスクリプトです。
"""
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

def main():
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="FDUNGSDQHN",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

    LLM = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1"
    )

    template = """以下のcontextに基づいて回答して下さい。

    <context>
    {context}
    </context>

    質問: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | output_parser
    )

    # output = chain.invoke({"question": "イベントストーミングの手順を教えてください。"})
    # output = chain.invoke("イベントストーミングの手順を教えてください。")
    output = chain.invoke("イミュータブルデータモデルのステップを教えてください。")
    print(output)


if __name__ == "__main__":
    main()
