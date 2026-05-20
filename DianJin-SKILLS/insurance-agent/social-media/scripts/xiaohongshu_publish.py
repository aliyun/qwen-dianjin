"""
小红书笔记自动发布工具
通过 Playwright 自动化操作小红书创作者后台，发布图文笔记。

前置条件:
    pip install playwright
    playwright install chromium

使用方式:
    # 首次使用：启动浏览器手动登录，保存登录状态
    python xiaohongshu_publish.py login

    # 发布笔记
    python xiaohongshu_publish.py publish --title "标题" --content "正文" --images img1.png img2.png --tags "保险知识" "养老规划"
"""

import argparse
import asyncio
import json
import os
import random
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("[错误] 请先安装 playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# 小红书创作者后台地址
CREATOR_URL = "https://creator.xiaohongshu.com/publish/publish"
AUTH_STATE_FILE = "xiaohongshu_auth.json"


async def random_delay(min_sec=0.5, max_sec=1.5):
    """随机延迟，模拟人工操作"""
    await asyncio.sleep(random.uniform(min_sec, max_sec))


async def login_and_save_state(config_dir):
    """启动浏览器让用户手动登录，自动检测登录成功后保存状态"""
    auth_path = os.path.join(config_dir, AUTH_STATE_FILE)

    print("=" * 50)
    print("  小红书登录 - 保存登录状态")
    print("=" * 50)
    print()
    print("即将打开浏览器，请手动完成登录操作。")
    print("登录成功后会自动检测并保存状态，无需手动操作。")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://creator.xiaohongshu.com/login")
        print("[等待] 请在浏览器中完成登录...")

        # 自动检测登录状态：轮询页面URL变化或检测登录后的元素
        max_wait = 300  # 最多等待5分钟
        for i in range(max_wait):
            await asyncio.sleep(1)
            current_url = page.url
            # 登录成功后会跳转离开login页面
            if "login" not in current_url and "creator.xiaohongshu.com" in current_url:
                print("[检测] 登录成功，正在保存状态...")
                await asyncio.sleep(3)  # 等待页面完全加载
                break
            # 也可能跳转到小红书主站
            if "www.xiaohongshu.com" in current_url and "login" not in current_url:
                print("[检测] 登录成功，正在跳转创作者中心...")
                await page.goto("https://creator.xiaohongshu.com/publish/publish")
                await asyncio.sleep(3)
                break
            if i > 0 and i % 30 == 0:
                print(f"  等待登录中... ({i}秒)")
        else:
            print("[超时] 等待登录超过5分钟，保存当前状态")

        await context.storage_state(path=auth_path)
        print(f"[成功] 登录状态已保存到: {auth_path}")

        await browser.close()

    return auth_path


async def publish_note(config_dir, title, content, image_paths, tags=None):
    """
    发布小红书图文笔记

    参数:
        config_dir: 配置文件目录（存放auth state）
        title: 笔记标题（最多20字）
        content: 笔记正文
        image_paths: 图片文件路径列表（1-18张）
        tags: 话题标签列表（可选）
    返回:
        成功返回 True，失败返回 False
    """
    auth_path = os.path.join(config_dir, AUTH_STATE_FILE)

    if not os.path.exists(auth_path):
        print(f"[错误] 登录状态文件不存在: {auth_path}")
        print("请先运行: python xiaohongshu_publish.py login")
        return False

    # 校验图片
    valid_images = []
    for img in image_paths:
        if not os.path.exists(img):
            print(f"[警告] 图片不存在，跳过: {img}")
            continue
        size = os.path.getsize(img)
        if size > 20 * 1024 * 1024:
            print(f"[警告] 图片超过20MB，跳过: {img}")
            continue
        valid_images.append(os.path.abspath(img))

    if not valid_images:
        print("[错误] 没有有效的图片可上传")
        return False

    if len(title) > 20:
        print(f"[警告] 标题超过20字，将截断: {title[:20]}...")
        title = title[:20]

    print("=" * 50)
    print("  小红书笔记发布")
    print("=" * 50)
    print(f"  标题: {title}")
    print(f"  图片: {len(valid_images)} 张")
    print(f"  标签: {tags if tags else '无'}")
    print("=" * 50)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=auth_path)
        page = await context.new_page()

        try:
            # 1. 打开创作者发布页
            print("\n[1/5] 打开小红书创作者中心...")
            await page.goto(CREATOR_URL, wait_until="networkidle", timeout=30000)
            await random_delay(2, 3)

            # 检查是否需要重新登录
            if "login" in page.url:
                print("[错误] 登录状态已过期，请重新运行: python xiaohongshu_publish.py login")
                return False

            # 1.5. 切换到「上传图文」标签页（默认在视频标签）
            js_click_image_tab = """() => {
                const tabs = document.querySelectorAll("div.creator-tab");
                for (let i = 0; i < tabs.length; i++) {
                    const text = tabs[i].textContent.trim();
                    if (text.includes("\u56fe\u6587") && !text.includes("\u56fe\u518c")) {
                        tabs[i].click();
                        return "clicked_index_" + i;
                    }
                }
                if (tabs.length >= 2) {
                    tabs[1].click();
                    return "clicked_fallback_1";
                }
                return "no_tabs";
            }"""
            tab_result = await page.evaluate(js_click_image_tab)
            await random_delay(2, 3)
            print(f"[成功] 已切换到图文发布模式 ({tab_result})")

            # 2. 上传图片（file input 是隐藏的，需要直接操作）
            print(f"\n[2/5] 上传图片（{len(valid_images)} 张）...")
            file_input = page.locator('input[type="file"]')
            await file_input.set_input_files(valid_images)
            await random_delay(3, 5)
            print("[成功] 图片上传完成")

            # 3. 填写标题
            print("\n[3/5] 填写标题...")
            title_input = await page.wait_for_selector(
                'input[placeholder*="标题"], input[class*="title"]',
                timeout=10000,
            )
            await title_input.click()
            await title_input.fill("")
            await title_input.type(title, delay=50)
            await random_delay()
            print(f"[成功] 标题已填写: {title}")

            # 4. 填写正文
            print("\n[4/5] 填写正文...")
            editor = await page.wait_for_selector(
                'div[contenteditable="true"], div[class*="editor"]',
                timeout=10000,
            )
            await editor.click()

            # 输入正文内容
            body_text = content
            if tags:
                tag_text = " " + " ".join(f"#{t}" for t in tags)
                body_text += tag_text

            await editor.type(body_text, delay=30)
            await random_delay(1, 2)
            print("[成功] 正文已填写")

            # 5. 点击发布
            print("\n[5/5] 发布笔记...")
            publish_btn = await page.wait_for_selector(
                'button:has-text("发布")', timeout=10000
            )
            await random_delay(1, 2)
            await publish_btn.click()

            # 等待发布结果
            try:
                await page.wait_for_selector(
                    'text=发布成功, text=已发布',
                    timeout=15000,
                )
                print("\n" + "=" * 50)
                print("  发布成功！")
                print("=" * 50)

                # 更新登录状态
                await context.storage_state(path=auth_path)
                return True

            except Exception:
                print("[警告] 未检测到发布成功提示，请在浏览器中确认发布状态")
                await random_delay(5, 8)
                # 仍然更新状态
                await context.storage_state(path=auth_path)
                return True

        except Exception as e:
            print(f"[错误] 发布过程出错: {e}")
            return False

        finally:
            await random_delay(2, 3)
            await browser.close()


async def publish_multiple_notes(config_dir, notes):
    """
    批量发布多条笔记（间隔发布，避免风控）

    参数:
        config_dir: 配置文件目录
        notes: 笔记列表，每个元素为 dict:
            {"title": str, "content": str, "images": [str], "tags": [str]}
    返回:
        每条笔记的发布结果列表
    """
    results = []
    total = len(notes)

    for i, note in enumerate(notes):
        print(f"\n{'='*60}")
        print(f"  发布第 {i+1}/{total} 条笔记")
        print(f"{'='*60}")

        success = await publish_note(
            config_dir,
            note["title"],
            note["content"],
            note["images"],
            note.get("tags"),
        )
        results.append({"index": i + 1, "title": note["title"], "success": success})

        if i < total - 1:
            wait_time = random.randint(60, 120)
            print(f"\n[等待] 为避免风控，等待 {wait_time} 秒后发布下一条...")
            await asyncio.sleep(wait_time)

    # 打印汇总
    print(f"\n{'='*60}")
    print("  发布结果汇总")
    print(f"{'='*60}")
    for r in results:
        status = "成功" if r["success"] else "失败"
        print(f"  第{r['index']}条 | {status} | {r['title']}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="小红书笔记自动发布工具")
    parser.add_argument(
        "--config-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="配置文件目录（默认为脚本所在目录）",
    )
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # login 子命令
    subparsers.add_parser("login", help="登录小红书并保存登录状态")

    # publish 子命令
    pub_parser = subparsers.add_parser("publish", help="发布一条笔记")
    pub_parser.add_argument("--title", required=True, help="笔记标题（最多20字）")
    pub_parser.add_argument("--content", required=True, help="笔记正文")
    pub_parser.add_argument(
        "--images", nargs="+", required=True, help="图片路径（1-18张）"
    )
    pub_parser.add_argument("--tags", nargs="*", help="话题标签")

    # batch 子命令
    batch_parser = subparsers.add_parser("batch", help="批量发布（从JSON文件读取）")
    batch_parser.add_argument("json_file", help="笔记数据JSON文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "login":
        asyncio.run(login_and_save_state(args.config_dir))

    elif args.command == "publish":
        success = asyncio.run(
            publish_note(args.config_dir, args.title, args.content, args.images, args.tags)
        )
        sys.exit(0 if success else 1)

    elif args.command == "batch":
        if not os.path.exists(args.json_file):
            print(f"[错误] JSON文件不存在: {args.json_file}")
            sys.exit(1)
        with open(args.json_file, "r", encoding="utf-8") as f:
            notes = json.load(f)
        results = asyncio.run(publish_multiple_notes(args.config_dir, notes))
        failed = sum(1 for r in results if not r["success"])
        sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
