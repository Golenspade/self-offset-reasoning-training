#!/usr/bin/env python3
"""
交互式贴吧爬虫 - 手动完成验证，自动爬取
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright


async def wait_for_user_verification(page):
    """等待用户手动完成验证码"""
    print("\n" + "=" * 60)
    print("⚠️  请在浏览器中手动完成以下操作：")
    print("   1. 完成滑块验证码（如果有）")
    print("   2. 登录百度账号（可选，但建议登录以减少验证码）")
    print("   3. 确保能看到贴吧帖子列表")
    print("   4. 完成后，在终端按 Enter 继续...")
    print("=" * 60 + "\n")

    # 等待用户按 Enter
    await asyncio.get_event_loop().run_in_executor(None, input, "按 Enter 继续 >>> ")

    # 检查是否成功
    title = await page.title()
    if "安全验证" in title or "登录" in title:
        print("⚠️  似乎还没有完成验证，但继续尝试...")
    else:
        print(f"✅ 页面标题: {title}")

    return True


async def extract_threads_from_page(page):
    """从当前页面提取帖子"""
    print("\n🔍 开始提取帖子...")

    # 等待页面加载
    await asyncio.sleep(2)

    # 尝试多种选择器
    thread_selectors = [
        'li[class*="thread"]',
        "li.j_thread_list",
        ".threadlist_lz",
    ]

    threads = []
    for selector in thread_selectors:
        threads = await page.query_selector_all(selector)
        if threads:
            print(f"✅ 使用选择器 '{selector}' 找到 {len(threads)} 个帖子")
            break

    if not threads:
        print("❌ 未找到帖子列表！")
        return []

    judgments = []

    for i, thread in enumerate(threads[:20]):  # 每页最多取20个
        try:
            # 提取标题
            title_elem = await thread.query_selector("a.j_th_tit")
            if not title_elem:
                title_elem = await thread.query_selector('a[href*="/p/"]')

            if not title_elem:
                continue

            title = await title_elem.inner_text()
            title = title.strip()

            if not title:
                continue

            # 过滤：只要包含判断性关键词的帖子
            judgment_keywords = [
                "吗",
                "？",
                "能",
                "会",
                "是",
                "不",
                "没",
                "别",
                "太",
                "怎么",
                "为什么",
                "如何",
                "有",
                "要",
            ]
            if not any(kw in title for kw in judgment_keywords):
                continue

            # 提取作者
            author_elem = await thread.query_selector(".tb_icon_author")
            if not author_elem:
                author_elem = await thread.query_selector('span[class*="author"]')
            author = await author_elem.inner_text() if author_elem else "匿名"

            # 提取回复数
            reply_elem = await thread.query_selector(".threadlist_rep_num")
            if not reply_elem:
                reply_elem = await thread.query_selector('span[class*="reply"]')
            replies = await reply_elem.inner_text() if reply_elem else "0"

            judgment = {
                "title": title,
                "content": f"来自贴吧的讨论：{title}",
                "author": author.strip(),
                "upvotes": int(replies.strip()) if replies.strip().isdigit() else 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "贴吧",
            }

            judgments.append(judgment)
            print(f"  [{i+1}] {title[:50]}... (回复: {replies})")

        except Exception as e:
            print(f"  ⚠️  提取第 {i+1} 个帖子失败: {e}")
            continue

    print(f"\n✅ 成功提取 {len(judgments)} 条判例")
    return judgments


async def crawl_tieba_interactive(tieba_name: str, max_pages: int = 3):
    """交互式爬取贴吧"""
    print(f"\n🎯 开始爬取贴吧: {tieba_name}")
    print(f"📄 计划爬取 {max_pages} 页\n")

    async with async_playwright() as p:
        # 启动浏览器（非 headless）
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,  # 减慢操作，更像人类
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="zh-CN",
        )

        page = await context.new_page()

        all_judgments = []

        for page_num in range(max_pages):
            url = (
                f"https://tieba.baidu.com/f?ie=utf-8&kw={tieba_name}&pn={page_num * 50}"
            )
            print(f"\n📄 第 {page_num + 1}/{max_pages} 页")
            print(f"🌐 访问: {url}")

            # 访问页面
            await page.goto(url, timeout=60000)
            await asyncio.sleep(2)

            # 第一次访问时，等待用户手动验证
            if page_num == 0:
                await wait_for_user_verification(page)

            # 提取帖子
            judgments = await extract_threads_from_page(page)
            all_judgments.extend(judgments)

            # 随机延时
            if page_num < max_pages - 1:
                delay = 3 + page_num  # 逐渐增加延时
                print(f"\n⏳ 等待 {delay} 秒后继续...")
                await asyncio.sleep(delay)

        # 保存结果
        output_file = "../data/raw/raw_judgments.json"
        print(f"\n💾 保存数据到: {output_file}")

        import os

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_judgments, f, ensure_ascii=False, indent=2)

        print(f"✅ 共爬取 {len(all_judgments)} 条判例")
        print(f"📊 数据已保存到: {output_file}")

        # 保持浏览器打开一会儿
        print("\n⏸️  浏览器将在 10 秒后关闭...")
        await asyncio.sleep(10)

        await browser.close()

        return all_judgments


async def main():
    """主函数"""
    # 爬取多个贴吧
    tieba_list = [
        "邦多利怀孕",
        "弱智",
        "抗压",
    ]

    all_data = []

    for tieba in tieba_list:
        try:
            data = await crawl_tieba_interactive(tieba, max_pages=2)
            all_data.extend(data)
        except Exception as e:
            print(f"❌ 爬取 {tieba} 失败: {e}")
            continue

    print(f"\n🎉 全部完成！共爬取 {len(all_data)} 条数据")


if __name__ == "__main__":
    asyncio.run(main())
