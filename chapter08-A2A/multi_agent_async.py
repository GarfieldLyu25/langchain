"""
 作者 lgf
 日期 2025/10/31
 多智能体异步协同示例 - 研究团队
 环境：conda环境tellme，使用dotenv管理API密钥
 
 场景：创建一个研究团队，包含：
 - 搜索专家：负责在线搜索信息
 - 数据分析师：负责数据分析和计算
 - 总结专家：负责整合信息并生成报告
"""
import os
import sys
import asyncio
import dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from datetime import datetime

# Windows 平台兼容性设置
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 加载环境变量
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")


# ==================== 安全打印工具 ====================

class SafePrinter:
    """安全的打印类，避免 I/O 操作错误"""

    @staticmethod
    def safe_print(*args, **kwargs):
        try:
            print(*args, **kwargs, flush=True)
        except (ValueError, OSError) as e:
            # 如果标准输出关闭，写入到文件
            try:
                with open("agent_log.txt", "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}]", *args, **kwargs, file=f)
            except:
                pass  # 忽略所有打印错误

# 创建全局打印函数
safe_print = SafePrinter.safe_print


# ==================== 工具定义 ====================

@tool
def google_search(query: str) -> str:
    """在Google上搜索实时信息、新闻、事实性问题"""
    try:
        search = SerpAPIWrapper()
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索出错: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持基本算术运算，如：2+2, 10*5, 100/4等"""
    try:
        # 安全的数学计算环境
        allowed_names = {
            "__builtins__": None,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        # 移除潜在危险字符
        safe_expr = expression.replace("__", "").replace("import", "")
        result = eval(safe_expr, allowed_names, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def analyze_data(data: str) -> str:
    """分析文本数据，提取关键信息和统计特征"""
    try:
        # 基本统计
        words = data.split()
        word_count = len(words)
        char_count = len(data)

        # 句子统计
        sentences = [s.strip() for s in data.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sentence_count = len(sentences)

        # 词频统计（前5个）
        from collections import Counter
        word_freq = Counter(word.lower() for word in words if len(word) > 3)
        top_words = word_freq.most_common(5)

        analysis = f"""
数据分析结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基本统计：
  • 总字数: {word_count}
  • 总字符数: {char_count}
  • 句子数: {sentence_count}
  • 平均词长: {char_count/word_count if word_count > 0 else 0:.2f}
  • 平均句长: {word_count/sentence_count if sentence_count > 0 else 0:.2f} 词

高频词汇 (Top 5)：
{chr(10).join([f"  • {word}: {count}次" for word, count in top_words])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return analysis
    except Exception as e:
        return f"分析错误: {str(e)}"


# ==================== 智能体定义 ====================

class ResearchAgent:
    """搜索专家智能体"""

    def __init__(self, name: str):
        self.name = name
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = [google_search]

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是{name}，一个专业的搜索专家。

你的职责：
- 使用google_search工具搜索最新、最准确的信息
- 识别关键信息并过滤无关内容
- 提供结构化、易于理解的搜索结果
- 确保信息的时效性和相关性

工作原则：
1. 精确搜索：使用恰当的关键词
2. 信息验证：确保来源可靠
3. 结果整理：以清晰的格式呈现"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # 关闭详细输出，避免回调冲突
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=60  # 60秒超时
        )

    async def ainvoke(self, task: str) -> Dict[str, Any]:
        """异步执行任务"""
        safe_print(f"\n🔍 [{self.name}] 开始搜索: {task}")
        start_time = datetime.now()

        try:
            # 使用 asyncio.to_thread 在单独的线程中运行
            result = await asyncio.to_thread(
                self.executor.invoke,
                {"input": task}
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            safe_print(f"✅ [{self.name}] 搜索完成 (耗时: {duration:.2f}s)")

            return {
                "agent": self.name,
                "task": task,
                "result": result['output'],
                "duration": duration,
                "timestamp": end_time,
                "success": True
            }
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            safe_print(f"❌ [{self.name}] 搜索失败: {str(e)}")
            return {
                "agent": self.name,
                "task": task,
                "result": f"任务执行失败: {str(e)}",
                "duration": duration,
                "timestamp": end_time,
                "success": False
            }


class AnalystAgent:
    """数据分析师智能体"""

    def __init__(self, name: str):
        self.name = name
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = [calculator, analyze_data]

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是{name}，一个专业的数据分析师。

你的职责：
- 对数据进行深度分析和统计
- 使用calculator进行精确的数学计算
- 使用analyze_data提取文本特征和模式
- 提供数据驱动的洞察和建议

工作原则：
1. 数据准确：确保计算和分析的准确性
2. 深度洞察：不仅报告数据，更要解释意义
3. 可视化思维：用清晰的方式呈现分析结果"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=60
        )

    async def ainvoke(self, task: str) -> Dict[str, Any]:
        """异步执行任务"""
        safe_print(f"\n📊 [{self.name}] 开始分析: {task}")
        start_time = datetime.now()

        try:
            result = await asyncio.to_thread(
                self.executor.invoke,
                {"input": task}
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            safe_print(f"✅ [{self.name}] 分析完成 (耗时: {duration:.2f}s)")

            return {
                "agent": self.name,
                "task": task,
                "result": result['output'],
                "duration": duration,
                "timestamp": end_time,
                "success": True
            }
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            safe_print(f"❌ [{self.name}] 分析失败: {str(e)}")
            return {
                "agent": self.name,
                "task": task,
                "result": f"任务执行失败: {str(e)}",
                "duration": duration,
                "timestamp": end_time,
                "success": False
            }


class SummarizerAgent:
    """总结专家智能体"""

    def __init__(self, name: str):
        self.name = name
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    async def ainvoke(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """异步整合所有结果"""
        safe_print(f"\n📝 [{self.name}] 开始整合结果...")
        start_time = datetime.now()

        # 过滤成功的结果
        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            return {
                "agent": self.name,
                "task": "整合所有结果",
                "result": "❌ 所有前置任务都失败了，无法生成报告",
                "duration": 0,
                "timestamp": datetime.now(),
                "success": False
            }

        combined_info = "\n\n".join([
            f"【{r['agent']}】\n任务: {r['task']}\n结果:\n{r['result']}\n耗时: {r['duration']:.2f}秒"
            for r in successful_results
        ])

        prompt = f"""请将以下多个智能体的工作结果整合成一份专业、结构化的研究报告：

{combined_info}

报告要求：
1. **执行摘要** (100字左右)
   - 简明扼要地概括核心发现
   
2. **关键发现** (分点列出3-5个要点)
   - 突出最重要的信息和洞察
   
3. **详细分析** (300-400字)
   - 深入分析各项发现的意义
   - 解释数据背后的趋势和模式
   
4. **结论和建议** (100-150字)
   - 提供可操作的建议
   - 指出未来发展方向

请使用专业的语言，保持逻辑清晰，使用适当的emoji和格式增强可读性。
"""

        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                prompt
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            safe_print(f"✅ [{self.name}] 整合完成 (耗时: {duration:.2f}s)")

            return {
                "agent": self.name,
                "task": "整合所有结果",
                "result": response.content,
                "duration": duration,
                "timestamp": end_time,
                "success": True
            }
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            safe_print(f"❌ [{self.name}] 整合失败: {str(e)}")
            return {
                "agent": self.name,
                "task": "整合所有结果",
                "result": f"整合失败: {str(e)}",
                "duration": duration,
                "timestamp": end_time,
                "success": False
            }


# ==================== 研究团队协调器 ====================

class ResearchTeam:
    """研究团队协调器 - 管理多个智能体协同工作"""

    def __init__(self):
        safe_print("\n🚀 正在初始化研究团队...")
        self.search_agent1 = ResearchAgent("搜索专家-1")
        self.search_agent2 = ResearchAgent("搜索专家-2")
        self.analyst = AnalystAgent("数据分析师")
        self.summarizer = SummarizerAgent("总结专家")
        safe_print("✅ 研究团队初始化完成\n")

    async def research(self, topic: str) -> Dict[str, Any]:
        """执行完整的研究流程"""
        safe_print("\n" + "="*80)
        safe_print(f"🎯 [启动] 研究团队开始工作 - 主题: {topic}")
        safe_print("="*80)

        overall_start = datetime.now()

        safe_print("\n📋 [阶段1] 并行信息搜索")
        safe_print("-" * 80)

        # 创建搜索任务
        search_tasks = [
            self.search_agent1.ainvoke(f"搜索关于'{topic}'的最新信息和发展动态"),
            self.search_agent2.ainvoke(f"搜索关于'{topic}'的技术细节、应用案例和实践经验"),
            self.analyst.ainvoke(f"分析'{topic}'相关的数据趋势、市场规模和发展前景")
        ]

        # 并行执行搜索任务
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                safe_print(f"⚠️  任务 {i+1} 出现异常: {str(result)}")
                processed_results.append({
                    "agent": f"Agent-{i+1}",
                    "task": "搜索任务",
                    "result": f"异常: {str(result)}",
                    "duration": 0,
                    "timestamp": datetime.now(),
                    "success": False
                })
            else:
                processed_results.append(result)

        safe_print("\n" + "="*80)
        safe_print("📋 [阶段2] 整合分析结果")
        safe_print("-" * 80)

        # 整合结果
        summary_result = await self.summarizer.ainvoke(processed_results)

        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds()

        # 统计信息
        safe_print("\n" + "="*80)
        safe_print("📊 [统计] 研究完成统计")
        safe_print("="*80)
        safe_print(f"⏱️  总耗时: {total_duration:.2f}秒")
        safe_print(f"👥 参与智能体: {len(processed_results) + 1}个")
        safe_print(f"✅ 成功任务: {sum(1 for r in processed_results if r.get('success', False))}/{len(processed_results)}")
        safe_print(f"📝 执行任务数: {len(processed_results) + 1}个")
        safe_print("\n各智能体耗时：")

        for result in processed_results:
            status = "✅" if result.get('success', False) else "❌"
            safe_print(f"  {status} {result['agent']}: {result['duration']:.2f}s")

        status = "✅" if summary_result.get('success', False) else "❌"
        safe_print(f"  {status} {summary_result['agent']}: {summary_result['duration']:.2f}s")

        safe_print("\n" + "="*80 + "\n")

        return {
            "topic": topic,
            "search_results": processed_results,
            "final_summary": summary_result,
            "total_duration": total_duration,
            "timestamp": overall_end,
            "success": summary_result.get('success', False)
        }


# ==================== 主函数 ====================

async def main():
    """主函数：演示多智能体异步协同"""
    safe_print("\n" + "="*80)
    safe_print(" 🤖 [系统] 多智能体异步协同研究系统")
    safe_print("="*80)

    try:
        # 创建研究团队
        team = ResearchTeam()

        # 预设研究主题
        test_topics = [
            "AI在医疗中的应用",
        ]

        for topic in test_topics:
            result = await team.research(topic)

            # 显示最终报告
            safe_print("\n" + "="*80)
            safe_print("📄 [报告] 最终研究报告")
            safe_print("="*80)
            safe_print(result['final_summary']['result'])
            safe_print("\n" + "="*80 + "\n")

            if topic != test_topics[-1]:
                input("\n⏸️  按回车继续下一个研究主题...\n")

        # 交互模式
        safe_print("\n" + "="*80)
        safe_print("💬 进入交互模式（输入'exit'或'退出'结束）")
        safe_print("="*80 + "\n")

        while True:
            try:
                user_topic = input("🔎 请输入研究主题: ").strip()

                if user_topic.lower() in ['exit', 'quit', '退出', 'q']:
                    safe_print("\n👋 再见！感谢使用多智能体研究系统\n")
                    break

                if not user_topic:
                    safe_print("⚠️  请输入有效的研究主题")
                    continue

                # 执行研究
                result = await team.research(user_topic)

                # 显示最终报告
                safe_print("\n" + "="*80)
                safe_print("📄 [报告] 最终研究报告")
                safe_print("="*80)
                safe_print(result['final_summary']['result'])
                safe_print("\n" + "="*80 + "\n")

            except KeyboardInterrupt:
                safe_print("\n\n⚠️  检测到中断信号")
                break
            except EOFError:
                safe_print("\n\n⚠️  输入结束")
                break

    except Exception as e:
        safe_print(f"\n❌ [错误] 系统错误: {str(e)}")
        import traceback
        safe_print("\n详细错误信息：")
        safe_print(traceback.format_exc())
    finally:
        safe_print("\n🛑 系统正在关闭...\n")
        # 确保所有异步任务完成
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        safe_print("\n\n👋 程序被用户中断")
    except Exception as e:
        safe_print(f"\n❌ 程序异常退出: {str(e)}")