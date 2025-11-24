#!/usr/bin/env python3
"""
赛博裁判长 - Playwright 爬虫模块
使用真实浏览器绕过反爬虫机制
"""

import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict
from playwright.async_api import async_playwright, Page, Browser
from datetime import datetime


class TiebaCrawler:
    """贴吧爬虫 - 使用 Playwright"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser: Browser = None
        self.judgments: List[Dict] = []
        
    async def random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """随机延时，模拟人类行为"""
        await asyncio.sleep(random.uniform(min_sec, max_sec))
    
    async def init_browser(self, playwright):
        """初始化浏览器"""
        print("🌐 启动浏览器...")
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',  # 隐藏自动化特征
            ]
        )

        # 创建上下文，设置真实的浏览器指纹
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='zh-CN',
            timezone_id='Asia/Shanghai',
        )

        # 加载保存的 Cookie（如果存在）
        import os
        import json
        cookie_file = 'baidu_cookies.json'
        if os.path.exists(cookie_file):
            print("🍪 加载已保存的 Cookie...")
            with open(cookie_file, 'r') as f:
                cookies = json.load(f)
                await context.add_cookies(cookies)
                print(f"✅ 已加载 {len(cookies)} 个 Cookie")

        # 注入反检测脚本
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        return context
    
    async def crawl_tieba_page(self, page: Page, tieba_name: str, page_num: int = 0) -> List[Dict]:
        """爬取单个贴吧页面"""
        url = f"https://tieba.baidu.com/f?ie=utf-8&kw={tieba_name}&pn={page_num}"
        
        print(f"📄 正在爬取: {tieba_name} 第 {page_num // 50 + 1} 页")
        
        try:
            # 访问页面
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await self.random_delay(0.5, 1.5)
            
            # 检查是否需要验证码
            page_title = await page.title()
            if "安全验证" in page_title or "验证" in page_title:
                print(f"  ⚠️  遇到验证码！请先运行 handle_captcha.py 获取 Cookie")
                return []

            # 等待内容加载（使用正确的选择器）
            await page.wait_for_selector('li[class*="thread"]', timeout=10000)

            # 提取帖子列表（使用正确的选择器）
            threads = await page.query_selector_all('li[class*="thread"]')
            
            judgments = []
            for thread in threads[:10]:  # 每页只取前10个
                try:
                    # 提取标题（使用正确的选择器）
                    title_elem = await thread.query_selector('a.j_th_tit')
                    if not title_elem:
                        continue
                    title = await title_elem.inner_text()
                    title = title.strip()

                    # 跳过空标题
                    if not title:
                        continue

                    # 提取作者（使用正确的选择器）
                    author_elem = await thread.query_selector('.tb_icon_author')
                    author = await author_elem.inner_text() if author_elem else 'unknown'
                    author = author.strip()
                    
                    # 提取回复数（作为热度指标）
                    reply_elem = await thread.query_selector('.threadlist_rep_num')
                    replies = await reply_elem.inner_text() if reply_elem else '0'
                    
                    # 简单过滤：标题中包含判断性词汇
                    judgment_keywords = ['吗', '？', '能', '会', '是', '不', '没', '别', '太']
                    if any(kw in title for kw in judgment_keywords):
                        judgments.append({
                            'title': title.strip(),
                            'content': f"{title.strip()}（需要进入详情页获取完整内容）",
                            'author': author,
                            'upvotes': int(replies) if replies.isdigit() else 0,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'source': url
                        })
                        
                except Exception as e:
                    print(f"  ⚠️  提取帖子失败: {e}")
                    continue
            
            print(f"  ✅ 提取到 {len(judgments)} 条数据")
            return judgments
            
        except Exception as e:
            print(f"  ❌ 页面爬取失败: {e}")
            return []
    
    async def crawl_tieba(self, tieba_name: str, max_pages: int = 3):
        """爬取整个贴吧"""
        async with async_playwright() as playwright:
            context = await self.init_browser(playwright)
            page = await context.new_page()
            
            for page_num in range(0, max_pages * 50, 50):
                judgments = await self.crawl_tieba_page(page, tieba_name, page_num)
                self.judgments.extend(judgments)
                
                # 随机延时，避免被封
                await self.random_delay(2.0, 4.0)
            
            await self.browser.close()
    
    async def crawl_multiple_tiebas(self, tieba_list: List[str], pages_per_tieba: int = 2):
        """爬取多个贴吧"""
        for tieba in tieba_list:
            print(f"\n🎯 开始爬取贴吧: {tieba}")
            await self.crawl_tieba(tieba, max_pages=pages_per_tieba)
            print(f"✅ {tieba} 爬取完成，当前总数据: {len(self.judgments)}")
    
    def save_to_json(self, output_file: Path):
        """保存为 JSON"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.judgments, f, ensure_ascii=False, indent=2)
        print(f"\n💾 数据已保存: {output_file}")
        print(f"📊 总计: {len(self.judgments)} 条")


async def main():
    """主函数"""
    crawler = TiebaCrawler(headless=True)
    
    # 目标贴吧列表（使用 URL 编码后的名称）
    tieba_list = [
        "邦多利怀孕",  # Playwright 会自动处理 URL 编码
        "弱智",
        "抗压",
    ]
    
    await crawler.crawl_multiple_tiebas(tieba_list, pages_per_tieba=2)
    
    # 保存数据
    output_file = Path(__file__).parent.parent / 'data' / 'raw' / 'raw_judgments.json'
    crawler.save_to_json(output_file)


if __name__ == '__main__':
    asyncio.run(main())

