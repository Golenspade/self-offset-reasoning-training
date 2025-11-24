#!/usr/bin/env python3
"""
赛博裁判长 - 核心插件
功能: 处理消息，调用 ReAct 循环，返回判决
"""

from nonebot import on_message, on_command
from nonebot.adapters.onebot.v11 import Bot, Event, GroupMessageEvent, PrivateMessageEvent
from nonebot.rule import to_me
from nonebot.log import logger
import asyncio
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.cerebras_client import CerebrasClient, PersonalityManager
from brain.react_loop import ReActAgent
from memory.sqlite_store import MemoryStore


# 初始化组件
try:
    llm_client = CerebrasClient()
    personality_manager = PersonalityManager(llm_client)
    memory_store = MemoryStore()
    react_agent = ReActAgent(llm_client, memory_store)
    logger.info("✅ 赛博裁判长组件初始化成功")
except Exception as e:
    logger.error(f"❌ 组件初始化失败: {e}")
    llm_client = None
    personality_manager = None
    memory_store = None
    react_agent = None


# 监听所有消息（用于记录历史）
message_logger = on_message(priority=100)

@message_logger.handle()
async def log_message(bot: Bot, event: Event):
    """记录所有消息到记忆库"""
    if not memory_store:
        return
    
    user_id = str(event.get_user_id())
    message = str(event.get_message())
    group_id = None
    
    if isinstance(event, GroupMessageEvent):
        group_id = str(event.group_id)
    
    try:
        await memory_store.save_message(user_id, message, group_id)
    except Exception as e:
        logger.error(f"保存消息失败: {e}")


# @裁判长 或 私聊触发判决
judge_matcher = on_message(rule=to_me(), priority=10)

@judge_matcher.handle()
async def handle_judge(bot: Bot, event: Event):
    """处理判决请求"""
    if not all([llm_client, react_agent, memory_store]):
        await judge_matcher.finish("⚠️ 裁判长系统未就绪，请稍后再试")
        return
    
    user_id = str(event.get_user_id())
    message = str(event.get_message()).strip()
    
    # 去除 @机器人 的部分
    message = message.replace(f"@{bot.self_id}", "").strip()
    
    if not message:
        await judge_matcher.finish("有事说事，别光@我")
        return
    
    try:
        # 显示"正在思考"
        await bot.send(event, "🤔 裁判长正在思考...")
        
        # 获取上下文
        context = ""
        if isinstance(event, GroupMessageEvent):
            context = f"群聊 {event.group_id}"
        
        # 使用 ReAct 循环生成回复
        response = await react_agent.think_and_act(
            user_message=message,
            user_id=user_id,
            context=context
        )
        
        # 保存判决记录
        await memory_store.save_judgment(
            user_id=user_id,
            case=message,
            verdict=response,
            personality="ReAct",
            group_id=str(event.group_id) if isinstance(event, GroupMessageEvent) else None
        )
        
        # 发送回复
        await judge_matcher.finish(response)
        
    except Exception as e:
        logger.error(f"判决失败: {e}")
        await judge_matcher.finish("⚠️ 裁判长罢工了，请稍后再试")


# 快速判决模式（不使用 ReAct，直接用多重人格）
quick_judge = on_command("快速判决", aliases={"qj", "快判"}, priority=5)

@quick_judge.handle()
async def handle_quick_judge(bot: Bot, event: Event):
    """快速判决（并发多重人格）"""
    if not all([personality_manager, memory_store]):
        await quick_judge.finish("⚠️ 裁判长系统未就绪")
        return
    
    user_id = str(event.get_user_id())
    message = str(event.get_message()).strip()
    
    # 去除命令部分
    for cmd in ["快速判决", "qj", "快判"]:
        message = message.replace(cmd, "").strip()
    
    if not message:
        await quick_judge.finish("请提供要判决的内容")
        return
    
    try:
        await bot.send(event, "⚡ 极速判决中...")
        
        # 并发多重人格
        result = await personality_manager.get_best_response(message)
        
        # 保存判决
        await memory_store.save_judgment(
            user_id=user_id,
            case=message,
            verdict=result['response'],
            personality=result['personality'],
            group_id=str(event.group_id) if isinstance(event, GroupMessageEvent) else None
        )
        
        # 发送回复（带人格标签）
        reply = f"[{result['personality']}] {result['response']}"
        await quick_judge.finish(reply)
        
    except Exception as e:
        logger.error(f"快速判决失败: {e}")
        await quick_judge.finish("⚠️ 判决失败")


# 查成分命令
check_profile = on_command("查成分", aliases={"成分", "profile"}, priority=5)

@check_profile.handle()
async def handle_check_profile(bot: Bot, event: Event):
    """查询用户成分"""
    if not memory_store:
        await check_profile.finish("⚠️ 记忆系统未就绪")
        return
    
    user_id = str(event.get_user_id())
    
    try:
        # 获取用户统计
        stats = await memory_store.get_user_stats(user_id)
        
        if not stats:
            await check_profile.finish("该用户无历史记录，鉴定为新人")
            return
        
        # 获取最近的判决
        judgments = await memory_store.get_user_judgments(user_id, limit=3)
        
        # 构建成分报告
        report = f"""📊 成分鉴定报告

用户ID: {user_id}
发言总数: {stats['total_messages']}
被判决次数: {stats['total_judgments']}
首次出现: {stats['first_seen']}
最后活跃: {stats['last_seen']}
"""
        
        if judgments:
            report += "\n📜 最近判决:\n"
            for i, j in enumerate(judgments, 1):
                report += f"{i}. [{j['personality']}] {j['verdict']}\n"
        
        await check_profile.finish(report)
        
    except Exception as e:
        logger.error(f"查成分失败: {e}")
        await check_profile.finish("⚠️ 查询失败")


# 帮助命令
help_cmd = on_command("裁判长帮助", aliases={"help", "帮助"}, priority=5)

@help_cmd.handle()
async def handle_help(bot: Bot, event: Event):
    """显示帮助信息"""
    help_text = """🤖 赛博裁判长使用指南

基础功能:
• @裁判长 [消息] - 深度判决（使用 ReAct 循环）
• /快速判决 [消息] - 极速判决（并发多重人格）
• /查成分 - 查询自己的历史记录

特性:
⚡ 亚秒级响应
🧠 具备记忆和推理能力
🎭 三种人格：暴躁老哥、理中客、阴阳人

示例:
@裁判长 4060能跑AI吗？
/快速判决 M3 Air能训练大模型吗？
/查成分
"""
    await help_cmd.finish(help_text)

