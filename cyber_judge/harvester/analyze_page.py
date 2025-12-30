#!/usr/bin/env python3
"""
分析贴吧页面结构，找到正确的选择器
"""

import asyncio
import json
from playwright.async_api import async_playwright


async def analyze_tieba_page():
    """分析贴吧页面结构"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        )

        # 加载 Cookie
        with open("baidu_cookies.json", "r") as f:
            cookies = json.load(f)
            await context.add_cookies(cookies)
            print(f"✅ 已加载 {len(cookies)} 个 Cookie")

        page = await context.new_page()

        url = "https://tieba.baidu.com/f?ie=utf-8&kw=邦多利怀孕"
        print(f"🌐 访问: {url}")

        await page.goto(url, timeout=60000)
        await asyncio.sleep(3)

        title = await page.title()
        print(f"📄 页面标题: {title}")

        if "安全验证" in title:
            print("❌ 仍然需要验证码，Cookie 可能已过期")
            return

        print("\n🔍 分析页面结构...")

        # 尝试各种可能的选择器
        selectors = {
            "帖子列表项": [
                'li[class*="thread"]',
                "li.j_thread_list",
                ".threadlist_lz",
                ".threadlist_bright",
                "ul#thread_list li",
                "li[data-field]",
            ],
            "帖子标题": [
                "a.j_th_tit",
                ".threadlist_title a",
                'a[href*="/p/"]',
                ".ti_title a",
                "div.threadlist_title a",
            ],
            "作者": [
                ".tb_icon_author",
                ".threadlist_author",
                "span.tb_icon_author",
                'span[class*="author"]',
            ],
            "回复数": [
                ".threadlist_rep_num",
                ".tb_icon_reply_num",
                "span.threadlist_rep_num",
            ],
        }

        results = {}

        for category, selector_list in selectors.items():
            print(f"\n📦 {category}:")
            for selector in selector_list:
                try:
                    elements = await page.query_selector_all(selector)
                    count = len(elements)

                    if count > 0:
                        print(f"  ✅ {selector}: {count} 个")

                        # 获取前3个元素的文本
                        samples = []
                        for elem in elements[:3]:
                            try:
                                text = await elem.inner_text()
                                text = text.strip()[:80]
                                if text:
                                    samples.append(text)
                            except:
                                pass

                        if samples:
                            results[category] = {
                                "selector": selector,
                                "count": count,
                                "samples": samples,
                            }
                            for i, sample in enumerate(samples):
                                print(f"      [{i}] {sample}")
                            break  # 找到一个就够了
                    else:
                        print(f"  ❌ {selector}: 0 个")
                except Exception as e:
                    print(f"  ⚠️  {selector}: 错误 - {e}")

        # 保存结果
        with open("page_analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 分析结果已保存到 page_analysis.json")

        # 尝试提取一个完整的帖子数据
        print("\n🎯 尝试提取第一个帖子的完整数据...")

        if "帖子列表项" in results:
            thread_selector = results["帖子列表项"]["selector"]
            threads = await page.query_selector_all(thread_selector)

            if threads:
                first_thread = threads[0]

                # 尝试提取各种信息
                data = {}

                # 标题
                if "帖子标题" in results:
                    title_elem = await first_thread.query_selector(
                        results["帖子标题"]["selector"]
                    )
                    if title_elem:
                        data["title"] = await title_elem.inner_text()
                        data["link"] = await title_elem.get_attribute("href")

                # 作者
                if "作者" in results:
                    author_elem = await first_thread.query_selector(
                        results["作者"]["selector"]
                    )
                    if author_elem:
                        data["author"] = await author_elem.inner_text()

                # 回复数
                if "回复数" in results:
                    reply_elem = await first_thread.query_selector(
                        results["回复数"]["selector"]
                    )
                    if reply_elem:
                        data["replies"] = await reply_elem.inner_text()

                print("\n📝 提取的数据:")
                for key, value in data.items():
                    print(f"  {key}: {value}")

        print("\n⏸️  浏览器将保持打开 30 秒，可以手动检查...")
        await asyncio.sleep(30)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(analyze_tieba_page())
