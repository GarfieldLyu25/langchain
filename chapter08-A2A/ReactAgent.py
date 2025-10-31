"""
 作者 lgf
 日期 2025/10/30
"""
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph,END
from typing import TypedDict,List
import os
import dotenv
from typing import Annotated
import operator
# 1. 定义工具
def search_tool(query: str):
    return f"搜索结果: {query} 的相关信息..."


def calculator_tool(expression: str):
    return eval(expression)


tools = [
    Tool(name="Search",func=search_tool,description="可以搜索查询互联网上的信息"),
    Tool(name="Calculator",func=calculator_tool,description="计算数学表达式")
]


# 2. 定义状态
class ReActState(TypedDict):
    messages: Annotated[list,operator.add]
    iterations: int

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
# 3. Agent 节点
llm = ChatOpenAI(model="gpt-4o-mini")


def agent_node(state: ReActState):
    """Agent 思考和决策"""
    messages = state['messages']

    prompt = f"""
    你是一个 ReAct Agent。根据用户问题,决定:
    **格式要求：**
    Thought: [你的思考过程]
    Action: [工具名称，只能是 Search 或 Calculator]
    Action Input: [工具的输入，必须是纯文本或数学表达式，不要有任何注释]
    或者
    - Final Answer: 最终答案

    可用工具: {[t.name for t in tools]}

    对话历史: {messages}
    """

    response = llm.invoke(prompt)

    return {
        "messages": [{"role": "assistant","content": response.content}],
        "iterations": state['iterations'] + 1
    }


def tool_node(state: ReActState):
    """执行工具"""
    last_message = state['messages'][-1]['content']

    # 解析工具调用 (简化版)
    if "Action: Search" in last_message:
        action_input = last_message.split("Action Input:")[-1].strip()
        result = search_tool(action_input)
    elif "Action: Calculator" in last_message:
        action_input = last_message.split("Action Input:")[-1].strip()
        result = calculator_tool(action_input)
    else:
        result = "无法识别的工具"

    return {
        "messages": [{"role": "tool","content": f"Observation: {result}"}]
    }


# 4. 路由逻辑
def should_continue(state: ReActState):
    last_message = state['messages'][-1]['content']

    if "Final Answer" in last_message:
        return "end"
    elif state['iterations'] >= 5:
        return "end"
    else:
        return "continue"


# 5. 构建图
workflow = StateGraph(ReActState)

workflow.add_node("agent",agent_node)
workflow.add_node("tools",tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools","agent")

app = workflow.compile()

# 6. 运行
result = app.invoke({
    "messages": [{"role": "user","content": "搜索 2025.10.31的新闻并计算 2**2+10"}],
    "iterations": 0
})

for msg in result['messages']:
    print(f"{msg['role']}: {msg['content']}\n")