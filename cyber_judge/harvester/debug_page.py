#!/usr/bin/env python3
"""
调试脚本 - 查看贴吧页面结构
"""

import asyncio
from playwright.async_api import async_playwright


async def debug_tieba():
    """调试贴吧页面"""
    async with async_playwright() as p:
        # 启动浏览器（非 headless 模式，可以看到）
        browser = await p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='zh-CN',
        )
        
        # 反检测
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        # 访问贴吧
        url = "https://tieba.baidu.com/f?ie=utf-8&kw=邦多利怀孕"
        print(f"🌐 正在访问: {url}")

        try:
            await page.goto(url, timeout=60000)
            print("✅ 页面加载完成")

            # 等待一下让页面完全渲染
            await asyncio.sleep(3)

            # 检查是否有验证码
            title = await page.title()
            if "安全验证" in title or "验证" in title:
                print("⚠️  检测到验证码页面！")
                print("💡 请手动完成验证码，脚本将等待 60 秒...")

                # 等待用户手动完成验证码
                for i in range(60):
                    await asyncio.sleep(1)
                    new_title = await page.title()
                    if "安全验证" not in new_title and "验证" not in new_title:
                        print(f"✅ 验证码已通过！新标题: {new_title}")
                        break
                    if i % 10 == 0:
                        print(f"   等待中... ({60-i}秒)")

                # 再等待页面加载
                await asyncio.sleep(3)
            
            # 获取页面标题
            title = await page.title()
            print(f"📄 页面标题: {title}")
            
            # 尝试不同的选择器
            selectors_to_try = [
                '.threadlist_title',
                '.threadlist_title a',
                '.j_thread_list',
                '.threadlist',
                'li.j_thread_list',
                '#thread_list li',
                '.t_con',
                '.threadlist_lz',
                'a.j_th_tit',
            ]
            
            print("\n🔍 测试选择器:")
            for selector in selectors_to_try:
                try:
                    elements = await page.query_selector_all(selector)
                    print(f"  {selector}: 找到 {len(elements)} 个元素")
                    
                    if len(elements) > 0 and len(elements) < 50:
                        # 打印前3个元素的文本
                        for i, elem in enumerate(elements[:3]):
                            try:
                                text = await elem.inner_text()
                                text = text.strip()[:50]  # 只显示前50个字符
                                print(f"    [{i}] {text}")
                            except:
                                pass
                except Exception as e:
                    print(f"  {selector}: 错误 - {e}")
            
            # 保存页面截图
            screenshot_path = "debug_tieba_page.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"\n📸 截图已保存: {screenshot_path}")
            
            # 保存 HTML
            html = await page.content()
            with open("debug_tieba_page.html", "w", encoding="utf-8") as f:
                f.write(html)
            print(f"💾 HTML 已保存: debug_tieba_page.html")
            
            # 保持浏览器打开，方便手动检查
            print("\n⏸️  浏览器将保持打开 30 秒，你可以手动检查页面...")
            print("   按 Ctrl+C 可以提前结束")
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        finally:
            await browser.close()
            print("\n✅ 调试完成")


if __name__ == '__main__':
    asyncio.run(debug_tieba())

