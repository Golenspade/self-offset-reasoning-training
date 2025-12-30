#!/usr/bin/env python3
"""
处理百度验证码的脚本
"""

import asyncio
from playwright.async_api import async_playwright


async def handle_baidu_captcha():
    """处理百度滑块验证码"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=100,  # 减慢操作速度，更像人类
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        page = await context.new_page()

        url = "https://tieba.baidu.com/f?ie=utf-8&kw=邦多利怀孕"
        print(f"🌐 访问: {url}")

        await page.goto(url, timeout=60000)
        await asyncio.sleep(2)

        # 检查是否有验证码
        title = await page.title()
        print(f"📄 页面标题: {title}")

        if "安全验证" in title:
            print("\n⚠️  检测到验证码！")
            print("🔍 查找验证码元素...")

            # 尝试查找 iframe
            frames = page.frames
            print(f"📦 页面有 {len(frames)} 个 frame")

            for i, frame in enumerate(frames):
                print(f"  Frame {i}: {frame.url[:100]}")

            # 等待验证码容器加载
            try:
                # 查找滑块容器
                slider_selectors = [
                    ".b0b2aae5ff",  # 从 HTML 中看到的滑块类名
                    'div[class*="slider"]',
                    'div[class*="slide"]',
                    "canvas",
                ]

                for selector in slider_selectors:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        print(f"✅ 找到元素: {selector} ({len(elements)} 个)")

                # 尝试点击并拖动滑块
                print("\n🎯 尝试拖动滑块...")

                # 查找滑块按钮
                slider_button = await page.query_selector(".b0b2aae5ff")

                if slider_button:
                    print("✅ 找到滑块按钮")

                    # 获取滑块位置
                    box = await slider_button.bounding_box()
                    if box:
                        print(
                            f"📍 滑块位置: x={box['x']}, y={box['y']}, width={box['width']}"
                        )

                        # 模拟人类拖动：不是直线，而是曲线
                        start_x = box["x"] + box["width"] / 2
                        start_y = box["y"] + box["height"] / 2

                        # 移动到滑块
                        await page.mouse.move(start_x, start_y)
                        await asyncio.sleep(0.2)

                        # 按下鼠标
                        await page.mouse.down()
                        await asyncio.sleep(0.1)

                        # 模拟人类拖动轨迹（带抖动）
                        distance = 280  # 大概的滑动距离
                        steps = 30

                        for i in range(steps):
                            # 计算当前位置（带随机抖动）
                            progress = i / steps
                            # 使用缓动函数（先快后慢）
                            eased_progress = 1 - (1 - progress) ** 2

                            current_x = start_x + distance * eased_progress
                            # 添加随机抖动
                            jitter_y = start_y + ((-1) ** i) * (2 if i % 3 == 0 else 1)

                            await page.mouse.move(current_x, jitter_y)
                            await asyncio.sleep(0.01 + 0.01 * (i % 3))  # 随机延时

                        # 释放鼠标
                        await asyncio.sleep(0.2)
                        await page.mouse.up()

                        print("✅ 滑块拖动完成")

                        # 等待验证结果
                        await asyncio.sleep(3)

                        # 检查是否通过
                        new_title = await page.title()
                        if "安全验证" not in new_title:
                            print(f"🎉 验证通过！新标题: {new_title}")
                        else:
                            print("❌ 验证失败，可能需要重试")

                else:
                    print("❌ 未找到滑块按钮")
                    print("\n💡 请手动完成验证，脚本将等待...")

                    # 等待用户手动完成
                    for i in range(60):
                        await asyncio.sleep(1)
                        try:
                            new_title = await page.title()
                            if "安全验证" not in new_title:
                                print(f"\n✅ 验证通过！")
                                break
                        except:
                            pass

                        if i % 10 == 0 and i > 0:
                            print(f"   等待中... ({60-i}秒)")

            except Exception as e:
                print(f"❌ 错误: {e}")

        # 验证通过后，查看页面结构
        await asyncio.sleep(2)
        final_title = await page.title()
        print(f"\n📄 最终标题: {final_title}")

        if "安全验证" not in final_title:
            print("\n🔍 查找帖子列表...")

            # 保存 Cookie 供后续使用
            cookies = await context.cookies()
            print(f"\n🍪 获取到 {len(cookies)} 个 Cookie")

            import json

            with open("baidu_cookies.json", "w") as f:
                json.dump(cookies, f, indent=2)
            print("💾 Cookie 已保存到 baidu_cookies.json")

            # 截图
            await page.screenshot(path="after_captcha.png", full_page=True)
            print("📸 截图已保存: after_captcha.png")

        print("\n⏸️  浏览器将保持打开 30 秒...")
        await asyncio.sleep(30)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(handle_baidu_captcha())
