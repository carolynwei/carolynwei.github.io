import json
import math
import re
from pathlib import Path


BLOG_BASE = Path("blogBase.json")
DOCS_DIR = Path("docs")


# 中文技术博客按约 300 字 / 分钟估算
CHARS_PER_MINUTE = 300


def get_reading_time(word_count):
    return max(1, math.ceil(word_count / CHARS_PER_MINUTE))


with BLOG_BASE.open("r", encoding="utf-8") as f:
    blog_data = json.load(f)


for key, post in blog_data.items():

    # Gmeek 的文章通常是 P1、P2、P3...
    if not key.startswith("P"):
        continue

    html_dir = post.get("htmlDir")
    word_count = post.get("wordCount")

    if not html_dir or word_count is None:
        continue

    html_path = Path(html_dir)

    if not html_path.exists():
        print(f"Skip: {html_path} does not exist")
        continue

    reading_time = get_reading_time(word_count)

    html = html_path.read_text(encoding="utf-8")

    badge = (
        f'<span class="reading-time">'
        f'⏱ 约 {reading_time} 分钟'
        f'</span>'
    )

    # 防止 Action 重复运行时反复插入
    html = re.sub(
        r'<span class="reading-time">.*?</span>',
        '',
        html,
        flags=re.DOTALL
    )

    # 找文章标题 </h1>，直接插在标题下面
    pattern = r'(</h1>)'

    replacement = (
        r'\1'
        '\n'
        f'<div class="reading-time-wrapper">{badge}</div>'
    )

    new_html, count = re.subn(
        pattern,
        replacement,
        html,
        count=1
    )

    if count == 0:
        print(f"Warning: cannot find h1 in {html_path}")
        continue

    html_path.write_text(new_html, encoding="utf-8")

    print(
        f"{html_path}: "
        f"{word_count} chars -> {reading_time} min"
    )
