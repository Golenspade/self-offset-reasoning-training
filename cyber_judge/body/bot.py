#!/usr/bin/env python3
"""
赛博裁判长 - NoneBot2 主程序
功能: 对接 QQ 协议，处理消息，调用大脑模块
"""

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter
from pathlib import Path


def init_bot():
    """初始化 Bot"""
    # 初始化 NoneBot
    nonebot.init()

    # 注册适配器
    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    # 加载插件
    nonebot.load_plugins(str(Path(__file__).parent / "plugins"))

    return nonebot.get_asgi()


if __name__ == "__main__":
    app = init_bot()
    nonebot.run()
