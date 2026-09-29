import json
import math
import sys
from pathlib import Path


# ============================================================
# 读取 Gmeek 根目录
#
# workflow 中会这样调用：
#
# python tool/add_reading_time.py /opt/Gmeek
#
# 所以 ROOT = /opt/Gmeek
# ============================================================

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

BLOG_BASE = ROOT / "blogBase.json"

# 中文技术文章阅读速度
CHARS_PER_MINUTE = 300


def get_reading_time(word_count):
    """
    根据文章字数估算阅读时间。
    至少显示 1 分钟。
    """
    return max(
        1,
        math.ceil(word_count / CHARS_PER_MINUTE)
    )


# ============================================================
# 读取 blogBase.json
# ============================================================

print(f"Reading blogBase.json from: {BLOG_BASE}")

if not BLOG_BASE.exists():
    raise FileNotFoundError(
        f"blogBase.json not found: {BLOG_BASE}"
    )


with BLOG_BASE.open("r", encoding="utf-8") as f:
    blog_data = json.load(f)


# ============================================================
# 重点：
#
# Gmeek 的文章不是：
#
# {
#   "P1": {...}
# }
#
# 而是：
#
# {
#   "postListJson": {
#       "P1": {...},
#       "P2": {...}
#   }
# }
#
# ============================================================

posts = blog_data.get("postListJson", {})

print(f"Found {len(posts)} posts")


# ============================================================
# 逐篇处理
# ============================================================

processed = 0


for key, post in posts.items():

    html_dir = post.get("htmlDir")
    word_count = post.get("wordCount", 0)
    post_title = post.get("postTitle", key)

    if not html_dir:
        print(f"Skip {key}: no htmlDir")
        continue

    # htmlDir 例如：
    #
    # docs/post/xxx.html
    #
    # ROOT 是 /opt/Gmeek
    #
    # 最终得到：
    #
    # /opt/Gmeek/docs/post/xxx.html

    html_path = ROOT / html_dir

    if not html_path.exists():
        print(
            f"Skip {key}: HTML does not exist: "
            f"{html_path}"
        )
        continue


    # ========================================================
    # 计算阅读时间
    # ========================================================

    reading_time = get_reading_time(word_count)

    print(
        f"Processing: {post_title} | "
        f"{word_count} chars | "
        f"{reading_time} min"
    )


    # ========================================================
    # 读取 HTML
    # ========================================================

    html = html_path.read_text(encoding="utf-8")


    # ========================================================
    # 防止重复插入
    #
    # 如果之前已经插入过，就先删除旧版本。
    # ========================================================

    import re

    html = re.sub(
        r'\s*<!-- reading-time-start -->.*?'
        r'<!-- reading-time-end -->\s*',
        '\n',
        html,
        flags=re.DOTALL
    )


    # ========================================================
    # 阅读时间 HTML
    #
    # 使用 inline style，
    # 不依赖 config.json 额外 CSS。
    # ========================================================

    reading_html = f"""
<!-- reading-time-start -->
<div
    class="reading-time"
    style="
        margin: -6px 0 22px 0;
        font-size: 14px;
        color: var(--color-fg-muted);
        opacity: 0.85;
    "
>
    ⏱ 约 {reading_time} 分钟读完
</div>
<!-- reading-time-end -->
"""


    # ========================================================
    # 插到正文开始之前
    #
    # Gmeek HTML 当前结构：
    #
    # </div>   <- header
    #
    # <div id="content">
    #
    # 所以直接在 content 前插入最稳定。
    # ========================================================

    marker = '<div id="content">'


    if marker not in html:
        print(
            f"Warning: cannot find content marker in "
            f"{html_path}"
        )
        continue


    html = html.replace(
        marker,
        reading_html + "\n" + marker,
        1
    )


    # ========================================================
    # 保存 HTML
    # ========================================================

    html_path.write_text(
        html,
        encoding="utf-8"
    )

    processed += 1

    print(
        f"✓ Added reading time: "
        f"{reading_time} min -> {html_path}"
    )


# ============================================================
# 最终结果
# ============================================================

print()
print("========================================")
print(f"Reading time processing finished.")
print(f"Processed posts: {processed}/{len(posts)}")
print("========================================")
