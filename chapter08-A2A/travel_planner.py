"""
 作者 lgf
 日期 2025/10/31
 MCP旅游攻略系统 - 集成高德地图API
 环境：conda环境tellme，使用dotenv管理API密钥
 未完成
 功能：
 - 高德地图POI搜索
 - 路线规划
 - 天气查询
 - 美食推荐
 - 酒店推荐
 - 生成完整旅游攻略
"""
import os
import asyncio
import dotenv
import json
import requests
from typing import List, Dict, Any, TypedDict
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# 加载环境变量
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY1")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

# 高德地图API密钥（需要在.env中配置）
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


# ==================== MCP工具定义 ====================
def _search_poi_internal(city: str,keyword: str,poi_type: str = "") -> str:
    """内部POI搜索函数（不是工具）"""
    try:
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            "key": AMAP_API_KEY,
            "keywords": keyword,
            "city": city,
            "types": poi_type,
            "offset": 10,
            "extensions": "all"
        }

        response = requests.get(url,params=params,timeout=10)
        data = response.json()

        if data['status'] == '1' and data['pois']:
            pois = []
            for poi in data['pois'][:5]:
                pois.append({
                    "名称": poi.get('name',''),
                    "地址": poi.get('address',''),
                    "类型": poi.get('type',''),
                    "评分": poi.get('biz_ext',{}).get('rating','暂无'),
                    "位置": poi.get('location',''),
                    "电话": poi.get('tel','暂无')
                })
            return json.dumps(pois,ensure_ascii=False,indent=2)
        else:
            return f"未找到相关POI"

    except Exception as e:
        return f"POI搜索出错: {str(e)}"
@tool
def search_poi(city: str, keyword: str, poi_type: str = "") -> str:
    """搜索城市中的POI（兴趣点）"""
    return _search_poi_internal(city, keyword, poi_type)

@tool
def get_route(origin_city: str, origin_poi: str, dest_city: str, dest_poi: str, mode: str = "公交") -> str:
    """
    查询两地之间的路线

    参数:
        origin_city: 出发城市
        origin_poi: 出发地点
        dest_city: 目的地城市
        dest_poi: 目的地点
        mode: 出行方式，可选"驾车"、"公交"、"步行"

    返回:
        路线信息
    """
    try:
        # 模拟路线规划（实际应用需要先获取POI坐标，然后调用路线规划API）
        route_info = {
            "出发": f"{origin_city} {origin_poi}",
            "到达": f"{dest_city} {dest_poi}",
            "出行方式": mode,
            "预计时间": "约2小时",
            "预计距离": "约50公里",
            "建议": f"建议选择{mode}出行，路线顺畅"
        }

        return json.dumps(route_info, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"路线规划出错: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    查询城市天气

    参数:
        city: 城市名称

    返回:
        天气信息
    """
    try:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "key": AMAP_API_KEY,
            "city": city,
            "extensions": "all"  # 返回未来3天预报
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data['status'] == '1' and data.get('forecasts'):
            forecast = data['forecasts'][0]
            weather_info = {
                "城市": forecast['city'],
                "发布时间": forecast['reporttime'],
                "未来天气": []
            }

            for cast in forecast['casts'][:3]:
                weather_info["未来天气"].append({
                    "日期": cast['date'],
                    "白天天气": cast['dayweather'],
                    "夜间天气": cast['nightweather'],
                    "白天温度": cast['daytemp'] + "°C",
                    "夜间温度": cast['nighttemp'] + "°C",
                    "风向": cast['daywind']
                })

            return json.dumps(weather_info, ensure_ascii=False, indent=2)
        else:
            return f"天气查询失败，可能原因：城市名称错误或API密钥无效"

    except Exception as e:
        return f"天气查询出错: {str(e)}"


@tool
def search_restaurant(city: str, cuisine_type: str = "", location: str = "") -> str:
    """搜索餐厅和美食"""
    keyword = f"{cuisine_type}餐厅" if cuisine_type else "美食"
    if location:
        keyword += f" {location}"
    return _search_poi_internal(city, keyword, "餐饮服务")


@tool
def search_hotel(city: str, area: str = "", price_range: str = "") -> str:
    """搜索酒店"""
    keyword = "酒店"
    if area:
        keyword += f" {area}"
    if price_range:
        keyword += f" {price_range}"
    return _search_poi_internal(city, keyword, "住宿服务")

# ==================== 智能体定义 ====================

class TravelAgent:
    """旅游规划智能体基类"""
    
    def __init__(self, name: str, role: str, tools: List):
        self.name = name
        self.role = role
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是{name}，{role}。
你的任务是为用户提供专业的旅游建议。
请使用提供的工具获取准确信息，并给出详细的建议。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def ainvoke(self, task: str) -> Dict[str, Any]:
        """异步执行任务"""
        print(f"\n{'='*60}")
        print(f"🤖 [{self.name}] 开始工作: {task}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            result = await asyncio.to_thread(
                self.executor.invoke,
                {"input": task}
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ [{self.name}] 完成任务 (耗时: {duration:.2f}s)\n")
            
            return {
                "agent": self.name,
                "role": self.role,
                "task": task,
                "result": result['output'],
                "duration": duration,
                "timestamp": end_time
            }
        except Exception as e:
            print(f"❌ [{self.name}] 任务失败: {str(e)}\n")
            return {
                "agent": self.name,
                "role": self.role,
                "task": task,
                "result": f"错误: {str(e)}",
                "duration": 0,
                "timestamp": datetime.now()
            }


# ==================== 工作流状态定义 ====================

class TravelPlanState(TypedDict):
    """旅游规划工作流状态"""
    destination: str
    duration: str
    preferences: str
    weather_info: str
    attractions: str
    restaurants: str
    hotels: str
    routes: str
    final_plan: str
    messages: List
    next_step: str


# ==================== MCP旅游规划系统 ====================

class MCPTravelPlanner:
    """MCP旅游规划系统 - 协调多个智能体"""
    
    def __init__(self):
        # 创建专业团队
        self.weather_agent = TravelAgent(
            "天气顾问",
            "负责查询天气信息，提供穿衣建议",
            [get_weather]
        )
        
        self.attraction_agent = TravelAgent(
            "景点专家",
            "负责推荐热门景点和旅游路线",
            [search_poi, get_route]
        )
        
        self.food_agent = TravelAgent(
            "美食顾问",
            "负责推荐当地美食和餐厅",
            [search_restaurant]
        )
        
        self.hotel_agent = TravelAgent(
            "住宿顾问",
            "负责推荐合适的酒店",
            [search_hotel]
        )
        
        self.planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    async def plan_travel(self, destination: str, duration: str, preferences: str = "") -> Dict[str, Any]:
        """
        规划完整的旅游行程
        
        参数:
            destination: 目的地城市
            duration: 旅游时长，如"3天2晚"
            preferences: 个人偏好，如"喜欢历史文化"、"追求美食体验"
        """
        print("\n" + "="*80)
        print(f"🗺️  MCP旅游规划系统启动")
        print("="*80)
        print(f"📍 目的地: {destination}")
        print(f"⏱️  时长: {duration}")
        print(f"💭 偏好: {preferences if preferences else '无特殊偏好'}")
        print("="*80)
        
        overall_start = datetime.now()
        
        # 第一阶段：并行收集信息
        print(f"\n{'🔍 阶段1：信息收集':^60}")
        print("="*80)
        
        tasks = [
            self.weather_agent.ainvoke(f"查询{destination}未来3天的天气情况"),
            self.attraction_agent.ainvoke(f"推荐{destination}最值得游览的5个景点，{preferences}"),
            self.food_agent.ainvoke(f"推荐{destination}最受欢迎的特色美食和餐厅"),
            self.hotel_agent.ainvoke(f"推荐{destination}性价比高的酒店")
        ]
        
        results = await asyncio.gather(*tasks)
        
        weather_info = results[0]['result']
        attractions = results[1]['result']
        restaurants = results[2]['result']
        hotels = results[3]['result']
        
        # 第二阶段：路线规划
        print(f"\n{'🗺️  阶段2：路线规划':^60}")
        print("="*80)
        
        route_task = f"基于以下景点信息，规划{duration}的游览路线：\n{attractions}"
        route_result = await self.attraction_agent.ainvoke(route_task)
        routes = route_result['result']
        
        # 第三阶段：生成完整攻略
        print(f"\n{'📝 阶段3：生成攻略':^60}")
        print("="*80)
        
        final_plan = await self._generate_final_plan(
            destination, duration, preferences,
            weather_info, attractions, restaurants, hotels, routes
        )
        
        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"✅ 旅游规划完成 (总耗时: {total_duration:.2f}秒)")
        print(f"{'='*80}\n")
        
        return {
            "destination": destination,
            "duration": duration,
            "weather": weather_info,
            "attractions": attractions,
            "restaurants": restaurants,
            "hotels": hotels,
            "routes": routes,
            "final_plan": final_plan,
            "total_duration": total_duration,
            "timestamp": overall_end
        }
    
    async def _generate_final_plan(self, destination, duration, preferences,
                                   weather, attractions, restaurants, hotels, routes):
        """生成最终旅游攻略"""
        print("📋 正在整合信息，生成完整攻略...\n")
        
        prompt = f"""请基于以下信息，生成一份完整的{destination} {duration}旅游攻略：

🌤️ **天气信息**：
{weather}

🏛️ **景点推荐**：
{attractions}

🍽️ **美食推荐**：
{restaurants}

🏨 **酒店推荐**：
{hotels}

🗺️ **路线规划**：
{routes}

**用户偏好**：{preferences if preferences else '无特殊偏好'}

请生成一份结构清晰、详细实用的旅游攻略，包含：
1. 📋 行程概览
2. 📅 每日详细安排（包括时间、景点、餐饮、住宿）
3. 💰 预算参考
4. ⚠️ 注意事项
5. 💡 实用建议

使用Markdown格式，让攻略美观易读。
"""
        
        response = await asyncio.to_thread(self.planner_llm.invoke, prompt)
        return response.content


# ==================== 主函数 ====================

async def main():
    """主函数"""
    print("\n" + "="*80)
    print(" 🗺️  MCP智能旅游规划系统 - 基于高德地图")
    print("="*80)
    print("\n📌 功能特色：")
    print("  ✅ 实时天气查询")
    print("  ✅ 智能景点推荐")
    print("  ✅ 美食餐厅搜索")
    print("  ✅ 酒店住宿建议")
    print("  ✅ 路线智能规划")
    print("  ✅ 多智能体协同")
    print("\n" + "="*80)

    
    try:
        planner = MCPTravelPlanner()
        
        # 测试案例
        test_cases = [
            {
                "destination": "北京",
                "duration": "3天2晚",
                "preferences": "对历史文化感兴趣，喜欢传统美食"
            },
            # 可以添加更多测试案例
        ]
        
        for case in test_cases:
            result = await planner.plan_travel(**case)
            
            # 打印最终攻略
            print("\n" + "="*80)
            print("📄 完整旅游攻略")
            print("="*80)
            print(result['final_plan'])
            print("\n" + "="*80)
            
            # 等待用户确认
            if case != test_cases[-1]:
                input("\n按回车继续下一个案例...\n")
        
        # 交互模式
        print("\n\n🎯 进入交互模式（输入'exit'退出）\n")
        while True:
            print("\n请输入旅游信息：")
            destination = input("  目的地城市: ").strip()
            
            if destination.lower() in ['exit', 'quit', '退出']:
                print("\n👋 感谢使用，祝您旅途愉快！\n")
                break
            
            if not destination:
                continue
            
            duration = input("  旅游时长 (如'3天2晚'): ").strip() or "3天2晚"
            preferences = input("  个人偏好 (可选): ").strip()
            
            result = await planner.plan_travel(destination, duration, preferences)
            
            print("\n" + "="*80)
            print("📄 完整旅游攻略")
            print("="*80)
            print(result['final_plan'])
            print("\n" + "="*80)
    
    except Exception as e:
        print(f"\n❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())