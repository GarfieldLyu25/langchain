"""
 作者 lgf
 日期 2025/10/31
 使用LangGraph的MCP旅游规划系统
 支持动态决策和用户交互
"""
import os
import asyncio
import dotenv
import json
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mcp_travel_planner import (
    search_poi, get_weather, search_restaurant, 
    search_hotel, get_route, AMAP_API_KEY
)

# ✅ 设置 UTF-8 编码
if sys.platform == 'win32':
    # 设置标准输出编码为 UTF-8
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")


# ==================== 增强的状态定义 ====================

class EnhancedTravelState(TypedDict):
    """增强的旅游规划状态"""
    # 用户输入
    destination: str
    duration: str
    budget: str
    preferences: List[str]
    travel_type: str  # "自由行"、"深度游"、"亲子游"等
    
    # 收集的信息
    weather_info: dict
    attractions: List[dict]
    restaurants: List[dict]
    hotels: List[dict]
    
    # 规划结果
    daily_schedule: List[dict]
    budget_breakdown: dict
    final_plan: str
    
    # 工作流控制
    messages: List
    current_step: str
    user_feedback: str
    revision_needed: bool


# ==================== LangGraph节点定义 ====================

class TravelPlanningGraph:
    """旅游规划工作流图"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [search_poi, get_weather, search_restaurant, search_hotel, get_route]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    async def collect_requirements(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """收集用户需求"""
        print(f"\n{'📋 收集需求':^60}")
        print("="*80)
        
        prompt = f"""分析用户的旅游需求：
目的地：{state['destination']}
时长：{state['duration']}
预算：{state['budget']}
偏好：{', '.join(state['preferences'])}
类型：{state['travel_type']}

请提供需求分析和建议。"""
        
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        print(f"✅ 需求分析完成\n")
        
        return {
            **state,
            "current_step": "weather_check",
            "messages": [HumanMessage(content=prompt), response]
        }
    
    async def check_weather(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """检查天气"""
        print(f"\n{'🌤️  天气查询':^60}")
        print("="*80)
        
        weather_result = get_weather.invoke({"city": state['destination']})
        weather_data = json.loads(weather_result) if weather_result else {}
        
        print(f"✅ 天气信息获取完成\n")
        
        return {
            **state,
            "weather_info": weather_data,
            "current_step": "find_attractions"
        }
    
    async def find_attractions(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """查找景点"""
        print(f"\n{'🏛️  景点搜索':^60}")
        print("="*80)
        
        # 根据偏好搜索不同类型的景点
        attraction_types = {
            "历史文化": "旅游景点|名胜古迹",
            "自然风光": "旅游景点|风景名胜",
            "现代都市": "商务住宅|购物服务",
            "亲子游乐": "休闲娱乐"
        }
        
        poi_type = attraction_types.get(state['preferences'][0] if state['preferences'] else '', "旅游景点")
        
        result = search_poi.invoke({
            "city": state['destination'],
            "keyword": "景点",
            "poi_type": poi_type
        })
        
        attractions = json.loads(result) if result else []
        
        print(f"✅ 找到 {len(attractions)} 个景点\n")
        
        return {
            **state,
            "attractions": attractions,
            "current_step": "find_restaurants"
        }
    
    async def find_restaurants(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """查找餐厅"""
        print(f"\n{'🍽️  美食搜索':^60}")
        print("="*80)
        
        result = search_restaurant.invoke({
            "city": state['destination'],
            "cuisine_type": "特色美食",
            "location": ""
        })
        
        restaurants = json.loads(result) if result else []
        
        print(f"✅ 找到 {len(restaurants)} 家餐厅\n")
        
        return {
            **state,
            "restaurants": restaurants,
            "current_step": "find_hotels"
        }
    
    async def find_hotels(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """查找酒店"""
        print(f"\n{'🏨 酒店搜索':^60}")
        print("="*80)
        
        # 根据预算选择酒店类型
        price_range = "经济型" if "经济" in state['budget'] else "舒适型"
        
        result = search_hotel.invoke({
            "city": state['destination'],
            "area": "市中心",
            "price_range": price_range
        })
        
        hotels = json.loads(result) if result else []
        
        print(f"✅ 找到 {len(hotels)} 家酒店\n")
        
        return {
            **state,
            "hotels": hotels,
            "current_step": "create_schedule"
        }
    
    async def create_schedule(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """创建行程安排"""
        print(f"\n{'📅 行程规划':^60}")
        print("="*80)
        
        days = int(state['duration'].split('天')[0])
        
        prompt = f"""基于以下信息，创建详细的{days}天行程安排：

景点：{json.dumps(state['attractions'], ensure_ascii=False)}
餐厅：{json.dumps(state['restaurants'], ensure_ascii=False)}
酒店：{json.dumps(state['hotels'], ensure_ascii=False)}
天气：{json.dumps(state['weather_info'], ensure_ascii=False)}

要求：
1. 每天安排合理（考虑距离和时间）
2. 包含早中晚餐安排
3. 考虑天气因素
4. 符合用户偏好：{', '.join(state['preferences'])}

返回JSON格式的每日安排。"""
        
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        print(f"✅ 行程安排完成\n")
        
        return {
            **state,
            "daily_schedule": [],  # 实际应解析LLM返回的JSON
            "current_step": "calculate_budget"
        }
    
    async def calculate_budget(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """计算预算"""
        print(f"\n{'💰 预算计算':^60}")
        print("="*80)
        
        days = int(state['duration'].split('天')[0])
        
        budget_breakdown = {
            "交通": "约500元/人",
            "住宿": f"约{200 * (days-1)}元/人",
            "餐饮": f"约{150 * days}元/人",
            "门票": "约300元/人",
            "其他": "约200元/人",
            "总计": f"约{500 + 200*(days-1) + 150*days + 300 + 200}元/人"
        }
        
        print(f"✅ 预算计算完成\n")
        
        return {
            **state,
            "budget_breakdown": budget_breakdown,
            "current_step": "generate_plan"
        }
    
    async def generate_final_plan(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """生成最终攻略"""
        print(f"\n{'📝 生成攻略':^60}")
        print("="*80)
        
        prompt = f"""生成完整的旅游攻略文档（Markdown格式）：

# {state['destination']} {state['duration']}旅游攻略

## 📋 行程概览
- 目的地：{state['destination']}
- 时长：{state['duration']}
- 预算：{state['budget']}
- 类型：{state['travel_type']}

## 🌤️ 天气预报
{json.dumps(state['weather_info'], ensure_ascii=False, indent=2)}

## 🏛️ 推荐景点
{json.dumps(state['attractions'], ensure_ascii=False, indent=2)}

## 🍽️ 美食推荐
{json.dumps(state['restaurants'], ensure_ascii=False, indent=2)}

## 🏨 住宿建议
{json.dumps(state['hotels'], ensure_ascii=False, indent=2)}

## 💰 预算明细
{json.dumps(state['budget_breakdown'], ensure_ascii=False, indent=2)}

## 📅 每日安排
（这里添加详细的每日行程）

## ⚠️ 注意事项
- 提前预订酒店和门票
- 关注天气变化
- 准备必要的证件和物品

## 💡 实用建议
- 推荐交通工具
- 最佳拍照地点
- 当地特色体验

请用专业、友好的语气完善这份攻略。"""
        
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        print(f"✅ 攻略生成完成\n")
        
        return {
            **state,
            "final_plan": response.content,
            "current_step": "review"
        }
    
    async def review_plan(self, state: EnhancedTravelState) -> EnhancedTravelState:
        """审核计划"""
        print(f"\n{'👀 计划审核':^60}")
        print("="*80)
        print("请查看生成的攻略...\n")
        
        # 这里可以添加人工审核或自动质量检查
        
        return {
            **state,
            "current_step": "end",
            "revision_needed": False
        }
    
    def route_next(self, state: EnhancedTravelState) -> Literal["end", "collect", "weather", "attractions", "restaurants", "hotels", "schedule", "budget", "generate", "review"]:
        """路由到下一个节点"""
        step_map = {
            "collect": "weather",
            "weather_check": "attractions",
            "find_attractions": "restaurants",
            "find_restaurants": "hotels",
            "find_hotels": "schedule",
            "create_schedule": "budget",
            "calculate_budget": "generate",
            "generate_plan": "review",
            "review": "end",
            "end": "end"
        }
        
        current = state.get("current_step", "collect")
        next_step = step_map.get(current, "end")
        
        return "end" if next_step == "end" else next_step


# ==================== 构建工作流 ====================

async def build_and_run_workflow(destination: str, duration: str, budget: str, 
                                 preferences: List[str], travel_type: str):
    """构建并运行工作流"""
    
    graph_builder = TravelPlanningGraph()
    
    # 创建状态图
    workflow = StateGraph(EnhancedTravelState)
    
    # 添加所有节点
    workflow.add_node("collect", graph_builder.collect_requirements)
    workflow.add_node("weather", graph_builder.check_weather)
    workflow.add_node("attractions", graph_builder.find_attractions)
    workflow.add_node("restaurants", graph_builder.find_restaurants)
    workflow.add_node("hotels", graph_builder.find_hotels)
    workflow.add_node("schedule", graph_builder.create_schedule)
    workflow.add_node("budget", graph_builder.calculate_budget)
    workflow.add_node("generate", graph_builder.generate_final_plan)
    workflow.add_node("review", graph_builder.review_plan)
    
    # 设置入口
    workflow.set_entry_point("collect")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "collect",
        graph_builder.route_next,
        {
            "weather": "weather",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "weather",
        graph_builder.route_next,
        {
            "attractions": "attractions",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "attractions",
        graph_builder.route_next,
        {
            "restaurants": "restaurants",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "restaurants",
        graph_builder.route_next,
        {
            "hotels": "hotels",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "hotels",
        graph_builder.route_next,
        {
            "schedule": "schedule",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "schedule",
        graph_builder.route_next,
        {
            "budget": "budget",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "budget",
        graph_builder.route_next,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate",
        graph_builder.route_next,
        {
            "review": "review",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "review",
        graph_builder.route_next,
        {
            "end": END
        }
    )
    
    # 编译
    app = workflow.compile()
    
    # 初始状态
    initial_state: EnhancedTravelState = {
        "destination": destination,
        "duration": duration,
        "budget": budget,
        "preferences": preferences,
        "travel_type": travel_type,
        "weather_info": {},
        "attractions": [],
        "restaurants": [],
        "hotels": [],
        "daily_schedule": [],
        "budget_breakdown": {},
        "final_plan": "",
        "messages": [],
        "current_step": "collect",
        "user_feedback": "",
        "revision_needed": False
    }
    
    # 运行工作流
    print("\n" + "="*80)
    print("🚀 LangGraph工作流启动")
    print("="*80)
    
    # ✅ 使用 ainvoke 替代 asyncio.to_thread(app.invoke, ...)
    final_state = await app.ainvoke(initial_state)
    
    return final_state


# ==================== 主函数 ====================

async def main():
    """主函数"""
    print("\n" + "="*80)
    print(" 🗺️  MCP智能旅游规划系统 (LangGraph版)")
    print("="*80)
    
    try:
        # 测试案例
        result = await build_and_run_workflow(
            destination="杭州",
            duration="3天2晚",
            budget="人均2000元",
            preferences=["自然风光", "传统美食"],
            travel_type="自由行"
        )
        
        print("\n" + "="*80)
        print("📄 最终旅游攻略")
        print("="*80)
        print(result['final_plan'])
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())