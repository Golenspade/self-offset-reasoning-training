#!/usr/bin/env python3
"""
深度爬取邦多利怀孕吧 - 包含回复内容
重点：获取判断、吐槽、评价的语料范式
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright


async def wait_for_user(page, message="按 Enter 继续"):
    """等待用户确认"""
    print(f"\n{'='*60}")
    print(f"⚠️  {message}")
    print(f"{'='*60}\n")
    await asyncio.get_event_loop().run_in_executor(None, input, ">>> ")
    return True


async def extract_thread_detail(context, thread_url):
    """在新标签页中提取帖子详情"""
    detail_page = None
    try:
        # 在新标签页打开
        detail_page = await context.new_page()
        await detail_page.goto(thread_url, timeout=30000)
        await asyncio.sleep(1.5)
        
        # 提取楼主内容
        op_content = ""
        op_elem = await detail_page.query_selector('.d_post_content')
        if op_elem:
            op_content = await op_elem.inner_text()

        # 提取回复（前20条）
        replies = []
        reply_elems = await detail_page.query_selector_all('.l_post')
        
        for i, reply_elem in enumerate(reply_elems[:20]):  # 只取前20条回复
            try:
                # 提取回复内容
                content_elem = await reply_elem.query_selector('.d_post_content')
                if not content_elem:
                    continue
                
                content = await content_elem.inner_text()
                content = content.strip()
                
                if not content or len(content) < 5:
                    continue
                
                # 提取作者
                author_elem = await reply_elem.query_selector('.p_author_name')
                author = await author_elem.inner_text() if author_elem else "匿名"
                
                replies.append({
                    'author': author.strip(),
                    'content': content,
                    'floor': i + 1
                })
                
            except Exception as e:
                continue
        
        return {
            'op_content': op_content.strip(),
            'replies': replies
        }

    except Exception as e:
        print(f"    ⚠️  提取详情失败: {e}")
        return None
    finally:
        # 关闭详情页标签
        if detail_page:
            await detail_page.close()


async def extract_threads_from_page(page, context, page_num, crawl_replies=True):
    """从当前页面提取帖子"""
    print(f"\n🔍 第 {page_num} 页 - 开始提取帖子...")
    
    await asyncio.sleep(2)
    
    # 查找帖子列表
    threads = await page.query_selector_all('li[class*="thread"]')
    
    if not threads:
        print("❌ 未找到帖子列表！")
        return []
    
    print(f"✅ 找到 {len(threads)} 个帖子")
    
    judgments = []
    
    for i, thread in enumerate(threads[:15]):  # 每页取15个
        try:
            # 提取标题
            title_elem = await thread.query_selector('a.j_th_tit')
            if not title_elem:
                continue
            
            title = await title_elem.inner_text()
            title = title.strip()
            
            if not title:
                continue
            
            # 获取帖子链接
            thread_link = await title_elem.get_attribute('href')
            if thread_link and not thread_link.startswith('http'):
                thread_link = f"https://tieba.baidu.com{thread_link}"
            
            # 提取作者
            author_elem = await thread.query_selector('.tb_icon_author')
            author = await author_elem.inner_text() if author_elem else "匿名"
            
            # 提取回复数
            reply_elem = await thread.query_selector('.threadlist_rep_num')
            reply_count = await reply_elem.inner_text() if reply_elem else "0"
            reply_count_int = int(reply_count.strip()) if reply_count.strip().isdigit() else 0
            
            print(f"  [{i+1}] {title[:40]}... (回复: {reply_count})")
            
            # 构建基础数据
            judgment = {
                'title': title,
                'author': author.strip(),
                'reply_count': reply_count_int,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': '邦多利怀孕吧',
                'url': thread_link
            }
            
            # 如果需要爬取回复，在新标签页打开详情
            if crawl_replies and thread_link and reply_count_int > 0:
                print(f"    🔗 在新标签页打开详情...")
                detail = await extract_thread_detail(context, thread_link)

                if detail:
                    judgment['content'] = detail['op_content']
                    judgment['replies'] = detail['replies']
                    print(f"    ✅ 获取到 {len(detail['replies'])} 条回复")
            else:
                judgment['content'] = title
                judgment['replies'] = []
            
            judgments.append(judgment)
            
        except Exception as e:
            print(f"  ⚠️  提取第 {i+1} 个帖子失败: {e}")
            continue
    
    print(f"\n✅ 本页成功提取 {len(judgments)} 条数据")
    return judgments


async def crawl_bandori_tieba(max_pages=5):
    """深度爬取邦多利怀孕吧"""
    print(f"\n🎯 开始深度爬取：邦多利怀孕吧")
    print(f"📄 计划爬取 {max_pages} 页（包含回复内容）\n")

    async with async_playwright() as p:
        # 启动浏览器（非 headless）
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
        )

        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='zh-CN',
        )

        page = await context.new_page()

        # 第一次访问
        url = "https://tieba.baidu.com/f?ie=utf-8&kw=邦多利怀孕"
        print(f"🌐 访问: {url}")
        await page.goto(url, timeout=60000)
        await asyncio.sleep(2)

        # 等待用户完成验证
        await wait_for_user(page, "请完成验证码/登录，确保能看到帖子列表后按 Enter")

        all_judgments = []

        # 爬取多页
        for page_num in range(1, max_pages + 1):
            if page_num > 1:
                # 翻页
                next_url = f"https://tieba.baidu.com/f?ie=utf-8&kw=邦多利怀孕&pn={(page_num-1) * 50}"
                print(f"\n📄 翻到第 {page_num} 页...")
                await page.goto(next_url, timeout=30000)
                await asyncio.sleep(2)

            # 提取数据（传入 context 用于打开新标签页）
            judgments = await extract_threads_from_page(page, context, page_num, crawl_replies=True)
            all_judgments.extend(judgments)

            # 统计
            total_replies = sum(len(j.get('replies', [])) for j in all_judgments)
            print(f"\n📊 当前进度: {len(all_judgments)} 个帖子, {total_replies} 条回复")

            # 延时
            if page_num < max_pages:
                delay = 3 + page_num
                print(f"⏳ 等待 {delay} 秒后继续...")
                await asyncio.sleep(delay)

        # 保存结果
        output_file = '../data/raw/bandori_judgments.json'
        print(f"\n💾 保存数据到: {output_file}")

        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_judgments, f, ensure_ascii=False, indent=2)

        # 统计信息
        total_replies = sum(len(j.get('replies', [])) for j in all_judgments)
        print(f"\n{'='*60}")
        print(f"🎉 爬取完成！")
        print(f"📊 统计:")
        print(f"   - 帖子数: {len(all_judgments)}")
        print(f"   - 回复数: {total_replies}")
        print(f"   - 总语料: {len(all_judgments) + total_replies} 条")
        print(f"   - 保存位置: {output_file}")
        print(f"{'='*60}\n")

        # 保持浏览器打开
        print("⏸️  浏览器将在 10 秒后关闭...")
        await asyncio.sleep(10)

        await browser.close()

        return all_judgments


async def main():
    """主函数"""
    await crawl_bandori_tieba(max_pages=5)


if __name__ == '__main__':
    asyncio.run(main())


