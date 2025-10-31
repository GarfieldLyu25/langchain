"""
 作者 lgf
 日期 2025/10/31
 使用LangGraph的多智能体协同示例 - 异步并发执行
 环境：conda环境tellme，使用dotenv管理API密钥
"""
import os
import sys
import asyncio
import dotenv
from typing import TypedDict, List, Sequence
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SerpAPIWrapper

# ==================== 安全打印函数 ====================

def safe_print(*args, **kwargs):
    """安全打印，防止 I/O 错误"""
    try:
        print(*args, **kwargs)
    except (ValueError, OSError):
        # 如果标准输出关闭，尝试写入日志文件
        try:
            with open("multi_agent_output.log", "a", encoding="utf-8") as f:
                print(*args, **kwargs, file=f)
        except:
            pass  # 静默失败，避免程序崩溃

# ==================== 🔥 Windows 编码完全修复 ====================

def fix_windows_encoding():
    """完全修复 Windows 控制台编码问题"""
    if sys.platform == 'win32':
        try:
            # 方法1: 使用 Windows API 设置控制台 UTF-8
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)  # UTF-8 输入
            kernel32.SetConsoleOutputCP(65001)  # UTF-8 输出
        except Exception as e:
            safe_print(f"[警告] Windows API 设置失败: {e}")

        # 方法2: 重定向标准输出到 UTF-8
        try:
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace'  # 🔥 关键：遇到无法编码字符时替换
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                errors='replace'
            )
        except Exception as e:
            safe_print(f"[警告] 输出重定向失败: {e}")

        # 方法3: 环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'


# 在导入其他模块之前修复编码
fix_windows_encoding()

# 加载环境变量
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")


# ==================== 🛡️ 安全的字符串处理 ====================

def safe_encode(text: str, max_length: int = None) -> str:
    """
    安全处理字符串，移除/替换无法在 GBK 中显示的字符

    Args:
        text: 原始文本
        max_length: 可选，截断长度

    Returns:
        安全的字符串
    """
    if not text:
        return ""

    try:
        # 尝试编码为 GBK，失败的字符替换为问号
        safe_text = text.encode('gbk', errors='replace').decode('gbk')
    except Exception:
        # 如果还是失败，使用 ASCII
        safe_text = text.encode('ascii', errors='replace').decode('ascii')

    # 可选截断
    if max_length and len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."

    return safe_text


def remove_emojis(text: str) -> str:
    """移除所有 emoji 表情"""
    import re
    # Emoji 的 Unicode 范围
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # 表情符号
        u"\U0001F300-\U0001F5FF"  # 符号和图标
        u"\U0001F680-\U0001F6FF"  # 交通和地图
        u"\U0001F1E0-\U0001F1FF"  # 旗帜
        u"\U00002702-\U000027B0"  # 杂项符号
        u"\U000024C2-\U0001F251"  # 包围字符
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# ==================== 🔍 真实 SerpAPI 工具 ====================

@tool
async def search_general(query: str) -> str:
    """
    通用搜索工具 - 使用真实 SerpAPI

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    try:
        safe_print(f"  [通用搜索] {safe_encode(query, 50)}")
        search = SerpAPIWrapper()

        # 异步执行搜索
        result = await asyncio.to_thread(search.run, query)

        # 🔥 关键：清理结果中的特殊字符
        safe_result = remove_emojis(result)
        safe_result = safe_encode(safe_result, max_length=1500)

        safe_print(f"  [完成] 搜索完成（{len(safe_result)} 字符）")
        return safe_result

    except Exception as e:
        error_msg = f"搜索出错: {safe_encode(str(e))}"
        safe_print(f"  [错误] {error_msg}")
        return error_msg


@tool
async def search_news(query: str) -> str:
    """
    新闻搜索工具 - 使用真实 SerpAPI

    Args:
        query: 新闻关键词

    Returns:
        新闻搜索结果
    """
    try:
        safe_print(f"  [新闻搜索] {safe_encode(query, 50)}")
        search = SerpAPIWrapper()

        # 异步执行新闻搜索
        result = await asyncio.to_thread(search.run, f"news: {query}")

        # 清理结果
        safe_result = remove_emojis(result)
        safe_result = safe_encode(safe_result, max_length=1500)

        safe_print(f"  [完成] 新闻搜索完成（{len(safe_result)} 字符）")
        return safe_result

    except Exception as e:
        error_msg = f"新闻搜索出错: {safe_encode(str(e))}"
        safe_print(f"  [错误] {error_msg}")
        return error_msg


@tool
async def fact_check(claim: str) -> str:
    """
    事实核查工具 - 使用真实 SerpAPI

    Args:
        claim: 需要核查的声明

    Returns:
        核查结果
    """
    try:
        safe_print(f"  [事实核查] {safe_encode(claim, 50)}")
        search = SerpAPIWrapper()

        result = await asyncio.to_thread(search.run, f"fact check: {claim}")

        # 清理结果
        safe_result = remove_emojis(result)
        safe_result = safe_encode(safe_result, max_length=1000)

        safe_print(f"  [完成] 核查完成（{len(safe_result)} 字符）")
        return safe_result

    except Exception as e:
        error_msg = f"核查出错: {safe_encode(str(e))}"
        safe_print((f"  [错误] {error_msg}"))
        return error_msg


# ==================== 状态定义 ====================

class NewsWorkflowState(TypedDict):
    """新闻工作流状态"""
    topic: str
    raw_news: str
    fact_checked: str
    edited_content: str
    final_report: str
    messages: Sequence[BaseMessage]
    next_agent: str
    search_results: List[str]  # 存储多次搜索结果


# ==================== 智能体节点（真实搜索版）====================

class AsyncNewsAgents:
    """异步新闻团队智能体 - 真实 SerpAPI 版本"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # 🔥 只使用真实的 SerpAPI 工具
        self.all_tools = [
            search_general,
            search_news,
            fact_check,
        ]
        self.llm_with_tools = self.llm.bind_tools(self.all_tools)

    async def search_node(self, state: NewsWorkflowState) -> NewsWorkflowState:
        """
        搜索节点 - 使用真实 SerpAPI 进行多角度搜索
        """
        safe_print(f"\n[搜索阶段] 主题: {safe_encode(state['topic'])}")
        start_time = datetime.now()

        try:
            # 🔥 并发执行多个真实搜索
            search_queries = [
                state['topic'],  # 通用搜索
                f"{state['topic']} latest news",  # 最新新闻
                f"{state['topic']} 2024 2025",  # 时间限定搜索
            ]

            safe_print(f"  正在并发执行 {len(search_queries)} 个搜索任务...")

            tasks = [
                search_general.ainvoke({"query": q})
                for q in search_queries
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 安全处理结果
            safe_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    safe_results.append(f"搜索 {i+1} 失败: {safe_encode(str(result))}")
                else:
                    safe_results.append(safe_encode(result))

            duration = (datetime.now() - start_time).total_seconds()
            safe_print(f"[搜索阶段] 完成（耗时: {duration:.2f}秒）")

            return {
                **state,
                "search_results": safe_results,
                "next_agent": "reporter"
            }
        except Exception as e:
            safe_print(f"[搜索阶段] 错误: {safe_encode(str(e))}")
            return {
                **state,
                "search_results": [f"搜索失败: {safe_encode(str(e))}"],
                "next_agent": "reporter"
            }

    async def reporter_node(self, state: NewsWorkflowState) -> NewsWorkflowState:
        """记者节点 - 整合搜索结果并生成新闻稿"""
        safe_print(f"\n[记者] 整合来自 {len(state.get('search_results', []))} 个搜索的信息")

        try:
            search_info = "\n\n".join([
                f"=== 搜索结果 {i+1} ===\n{result}"
                for i, result in enumerate(state.get('search_results', []))
            ])

            messages = [
                HumanMessage(content=f"""你是一名专业记者，以下是关于"{state['topic']}"的搜索结果：

{search_info}

请基于这些真实搜索结果，撰写一篇新闻报道草稿，包括：

1. **新闻摘要**（100-150字）
2. **关键事件**（3-5个要点）
3. **相关背景**
4. **最新动态**

要求：
- 基于搜索结果的真实信息
- 保持客观中立
- 引用具体来源
- 如果信息不足，请如实说明

如果需要补充信息，可以使用 search_news 工具进行新闻搜索。""")
            ]

            response = await self.llm_with_tools.ainvoke(messages)

            # 处理工具调用
            if response.tool_calls:
                safe_print(f"  记者正在调用 {len(response.tool_calls)} 个工具补充信息...")
                tool_node = ToolNode(self.all_tools)
                tool_results = await tool_node.ainvoke({"messages": [response]})

                final_response = await self.llm.ainvoke(
                    messages + [response] + tool_results["messages"]
                )
                raw_news = safe_encode(final_response.content)
            else:
                raw_news = safe_encode(response.content)

            safe_print(f"[记者] 新闻草稿完成（{len(raw_news)} 字符）")

            return {
                **state,
                "raw_news": raw_news,
                "messages": [*state.get("messages", []), *messages, response],
                "next_agent": "fact_checker"
            }
        except Exception as e:
            safe_print(f"[记者] 错误: {safe_encode(str(e))}")
            return {
                **state,
                "raw_news": f"新闻整合失败: {safe_encode(str(e))}",
                "next_agent": "fact_checker"
            }

    async def fact_checker_node(self, state: NewsWorkflowState) -> NewsWorkflowState:
        """事实核查节点 - 使用真实 fact_check 工具"""
        safe_print(f"\n[事实核查] 开始核查新闻内容")

        try:
            messages = [
                HumanMessage(content=f"""你是一名事实核查员，请核查以下新闻草稿中的关键声明：

{state['raw_news']}

任务：
1. 识别需要核查的关键声明（2-3个最重要的）
2. 使用 fact_check 工具验证这些声明
3. 提供核查报告

请明确指出：
- ✓ 已验证的事实
- ? 需要更多信息的声明
- ✗ 可能存在问题的内容""")
            ]

            response = await self.llm_with_tools.ainvoke(messages)

            # 处理工具调用
            if response.tool_calls:
                safe_print(f"  正在核查 {len(response.tool_calls)} 个声明...")
                tool_node = ToolNode(self.all_tools)
                tool_results = await tool_node.ainvoke({"messages": [response]})

                final_response = await self.llm.ainvoke(
                    messages + [response] + tool_results["messages"]
                )
                fact_checked = safe_encode(final_response.content)
            else:
                fact_checked = safe_encode(response.content)

            safe_print(f"[事实核查] 核查完成（{len(fact_checked)} 字符）")

            return {
                **state,
                "fact_checked": fact_checked,
                "messages": [*state.get("messages", []), response],
                "next_agent": "editor"
            }
        except Exception as e:
            safe_print(f"[事实核查] 错误: {safe_encode(str(e))}")
            return {
                **state,
                "fact_checked": "核查过程出错",
                "next_agent": "editor"
            }

    async def editor_node(self, state: NewsWorkflowState) -> NewsWorkflowState:
        """编辑节点"""
        safe_print(f"\n[编辑] 开始审核和优化内容")

        try:
            messages = [
                HumanMessage(content=f"""你是一名资深编辑，请审核以下内容：

**新闻草稿**：
{state['raw_news']}

**事实核查报告**：
{state['fact_checked']}

任务：
1. 根据事实核查结果修改草稿
2. 优化语言表达
3. 确保逻辑连贯
4. 保持客观中立

输出一份经过编辑的最终新闻稿。不需要使用工具。""")
            ]

            response = await self.llm.ainvoke(messages)
            edited_content = safe_encode(response.content)

            safe_print(f"[编辑] 内容审核完成（{len(edited_content)} 字符）")

            return {
                **state,
                "edited_content": edited_content,
                "messages": [*state.get("messages", []), response],
                "next_agent": "publisher"
            }
        except Exception as e:
            safe_print(f"[编辑] 错误: {safe_encode(str(e))}")
            return {
                **state,
                "edited_content": state['raw_news'],
                "next_agent": "publisher"
            }

    async def publisher_node(self, state: NewsWorkflowState) -> NewsWorkflowState:
        """发布者节点"""
        safe_print(f"\n[发布者] 生成最终报告")

        try:
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

            prompt = f"""你是一名专业的内容发布者，请将以下内容格式化为专业的新闻报道：

{state['edited_content']}

要求：
1. 使用清晰的Markdown格式
2. 包含标题、摘要、正文、结论
3. 发布时间：{current_time}
4. 主题：{state['topic']}

输出格式：
---
# 📰 [新闻标题]

**📅 发布时间**: {current_time}  
**🏷️ 主题**: {state['topic']}  
**🔍 信息来源**: 基于 SerpAPI 真实搜索 + 事实核查

---

## 📝 摘要
[摘要内容]

---

## 📊 正文

### 关键事件
[事件列表]

### 详细报道
[详细内容]

### 背景信息
[背景]

---

## 🎯 结论
[总结]

---

## ✅ 核查说明
{state['fact_checked'][:200]}...

---
"""

            response = await self.llm.ainvoke(prompt)
            final_report = safe_encode(response.content)

            safe_print(f"[发布者] 报告生成完成（{len(final_report)} 字符）")

            return {
                **state,
                "final_report": final_report,
                "next_agent": "end"
            }
        except Exception as e:
            safe_print(f"[发布者] 错误: {safe_encode(str(e))}")
            return {
                **state,
                "final_report": f"报告生成失败: {safe_encode(str(e))}",
                "next_agent": "end"
            }


# ==================== 路由函数 ====================

def route_next(state: NewsWorkflowState) -> str:
    """决定下一个执行的节点"""
    next_agent = state.get("next_agent", "search")
    return END if next_agent == "end" else next_agent


# ==================== 异步工作流 ====================

class AsyncNewsWorkflow:
    """异步新闻工作流 - 真实 SerpAPI 版本"""

    def __init__(self):
        safe_print("\n[初始化] 正在初始化异步新闻工作流（真实 SerpAPI 版本）...")
        self.agents = AsyncNewsAgents()

        workflow = StateGraph(NewsWorkflowState)

        # 添加节点
        workflow.add_node("search", self.agents.search_node)
        workflow.add_node("reporter", self.agents.reporter_node)
        workflow.add_node("fact_checker", self.agents.fact_checker_node)
        workflow.add_node("editor", self.agents.editor_node)
        workflow.add_node("publisher", self.agents.publisher_node)

        # 设置入口
        workflow.set_entry_point("search")

        # 添加边
        workflow.add_conditional_edges(
            "search",
            route_next,
            {"reporter": "reporter", END: END}
        )

        workflow.add_conditional_edges(
            "reporter",
            route_next,
            {"fact_checker": "fact_checker", END: END}
        )

        workflow.add_conditional_edges(
            "fact_checker",
            route_next,
            {"editor": "editor", END: END}
        )

        workflow.add_conditional_edges(
            "editor",
            route_next,
            {"publisher": "publisher", END: END}
        )

        workflow.add_conditional_edges(
            "publisher",
            route_next,
            {END: END}
        )

        self.app = workflow.compile()
        safe_print("[初始化] 异步工作流初始化完成\n")

    async def run(self, topic: str) -> NewsWorkflowState:
        """异步运行工作流"""
        safe_print("\n" + "="*80)
        safe_print(f"[启动] 异步新闻工作流启动")
        safe_print(f"[主题] {safe_encode(topic)}")
        safe_print(f"[工具] 真实 SerpAPI 搜索 + 事实核查")
        safe_print("="*80)

        start_time = datetime.now()

        initial_state: NewsWorkflowState = {
            "topic": topic,
            "raw_news": "",
            "fact_checked": "",
            "edited_content": "",
            "final_report": "",
            "messages": [],
            "next_agent": "search",
            "search_results": []
        }

        try:
            final_state = await self.app.ainvoke(initial_state)
            duration = (datetime.now() - start_time).total_seconds()

            safe_print("\n" + "="*80)
            safe_print(f"[完成] 工作流完成（总耗时: {duration:.2f}秒）")
            safe_print("="*80 + "\n")

            return final_state
        except Exception as e:
            safe_print(f"\n[错误] 工作流错误: {safe_encode(str(e))}")
            raise


# ==================== 主函数 ====================

async def main():
    """主函数"""
    safe_print("\n" + "="*80)
    safe_print(" 异步多智能体新闻工作流系统 (LangGraph)")
    safe_print(" 真实 SerpAPI 搜索 + 事实核查")
    safe_print(" Windows 编码安全版本")
    safe_print("="*80)

    try:
        workflow = AsyncNewsWorkflow()

        # 演示：并发处理多个主题
        safe_print("\n演示: 并发处理多个新闻主题")
        safe_print("-"*80)

        topics = [
            "吕国凡是谁",
            "Garfieldlyu 最近写了什么代码",
        ]

        safe_print(f"将并发处理 {len(topics)} 个主题...")

        tasks = [workflow.run(topic) for topic in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                safe_print(f"\n[错误] 报道 {i+1} 失败: {safe_encode(str(result))}")
            else:
                safe_print(f"\n[完成] 报道 {i+1}: {safe_encode(topics[i])}")
                safe_print("="*80)
                safe_print(result['final_report'])
                safe_print("\n" + "="*80 + "\n")

        # 交互模式
        safe_print("\n" + "="*80)
        safe_print("进入交互模式（输入'exit'或'退出'结束）")
        safe_print("="*80 + "\n")

        while True:
            try:
                user_input = input("请输入新闻主题: ").strip()

                if user_input.lower() in ['exit', 'quit', '退出', 'q']:
                    safe_print("\n再见！感谢使用异步新闻工作流系统\n")
                    break

                if not user_input:
                    safe_print("[警告] 请输入有效的主题")
                    continue

                result = await workflow.run(user_input)
                safe_print("\n最终新闻报道")
                safe_print("="*80)
                safe_print(result['final_report'])
                safe_print("\n" + "="*80 + "\n")

            except KeyboardInterrupt:
                safe_print("\n\n[警告] 检测到中断信号")
                break
            except EOFError:
                safe_print("\n\n[警告] 输入结束")
                break
            except Exception as e:
                safe_print(f"\n[错误] 错误: {safe_encode(str(e))}")
                continue

    except Exception as e:
        safe_print(f"\n[错误] 系统错误: {safe_encode(str(e))}")
        import traceback
        traceback.print_exc()
    finally:
        safe_print("\n系统正在关闭...\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[警告] 程序被用户中断")
    except Exception as e:
        print(f"\n[错误] 程序异常退出: {safe_encode(str(e))}")