"""
 作者 lgf
 日期 2025/10/31
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from typing import TypedDict,Annotated
import operator
import os
import dotenv

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")


# ===== 1. 定义工具 =====
@tool
def search_tool(query: str) -> str:
    """搜索互联网信息

    Args:
        query: 搜索关键词

    Returns:
        搜索结果
    """
    return f"🔍 搜索结果: 比亚迪今天股票320块"


@tool
def calculator_tool(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式，如 "2**2+10"

    Returns:
        计算结果
    """
    try:
        result = eval(expression,{"__builtins__": {}},{})
        return f"📊 计算结果: {expression} = {result}"
    except Exception as e:
        return f"❌ 计算错误: {str(e)}"


tools = [search_tool,calculator_tool]


# ===== 2. 定义状态 =====
class AgentState(TypedDict):
    messages: Annotated[list,operator.add]


# ===== 3. 初始化 LLM =====
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ===== 4. 定义节点 =====
def agent_node(state: AgentState):
    """Agent 决策节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """判断是否继续"""
    last_message = state["messages"][-1]
    if hasattr(last_message,'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


# ===== 5. 构建图 =====
workflow = StateGraph(AgentState)

workflow.add_node("agent",agent_node)
workflow.add_node("tools",ToolNode(tools))  # 👈 自动处理工具调用

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent",should_continue,{
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools","agent")

app = workflow.compile()

# ===== 6. 运行 =====
if __name__ == "__main__":
    print("🚀 启动 Agent\n")

    result = app.invoke({
        "messages": [HumanMessage(content="搜索 2025.10.31的新闻并计算 2**2+10")]
    })

    print("\n" + "=" * 60)
    print("💬 对话历史:")
    print("=" * 60)

    for i,msg in enumerate(result["messages"],1):
        role = msg.__class__.__name__

        if hasattr(msg,'content') and msg.content:
            print(f"\n[{i}] {role}:")
            print(f"    {msg.content}")

        if hasattr(msg,'tool_calls') and msg.tool_calls:
            print(f"\n    🔧 工具调用:")
            for tc in msg.tool_calls:
                print(f"       - {tc['name']}: {tc['args']}")