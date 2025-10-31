"""
 作者 lgf
 日期 2025/10/31
 使用Google搜索工具的简化版本（使用@tool装饰器）
"""
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper

# 加载环境变量
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")


# 1. 使用@tool装饰器创建Google搜索工具
@tool
def google_search(query: str) -> str:
    """
    在Google上搜索实时信息。
    
    参数:
        query: 搜索查询字符串
    
    返回:
        搜索结果摘要
    """
    try:
        search = SerpAPIWrapper()
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索出错: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。
    
    参数:
        expression: 要计算的数学表达式，如 "2+2" 或 "10*5"
    
    返回:
        计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 2. 创建Agent
def create_agent():
    """创建使用工具的Agent"""
    # 初始化LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 工具列表
    tools = [google_search, calculator]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# 3. 主函数
def main():
    """主函数：测试Agent"""
    print("="*50)
    print("Google搜索Agent启动（简化版）")
    print("="*50)
    
    agent_executor = create_agent()
    
    # 交互式问答
    while True:
        user_input = input("\n请输入问题（输入'exit'退出）: ")
        
        if user_input.lower() == 'exit':
            print("再见！")
            break
        
        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"\n答案: {result['output']}\n")
        except Exception as e:
            print(f"错误: {str(e)}\n")


if __name__ == "__main__":
    # 可以直接运行main()进行交互
    # 或者测试单个问题
    agent_executor = create_agent()
    
    # 测试示例
    test_query = "搜索LangChain是什么"
    print(f"测试问题: {test_query}\n")
    result = agent_executor.invoke({"input": test_query})
    print(f"\n最终答案: {result['output']}")
