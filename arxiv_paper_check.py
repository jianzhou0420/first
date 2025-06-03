#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import feedparser
import schedule
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 配置部分 —— 请根据实际情况修改
ARXIV_QUERY = "manipulation"
ARXIV_CATEGORY = "cs.RO"
ARXIV_API_URL = (
    "http://export.arxiv.org/api/query"
    f"?search_query=all:{ARXIV_QUERY}+AND+cat:{ARXIV_CATEGORY}"
    "&start=0"
    "&max_results=50"
    "&sortBy=submittedDate"
    "&sortOrder=descending"
)
DATA_FILE = "arxiv_manipulation.json"       # 上一次结果保存文件
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = "jianzhou0420@gmail.com"
EMAIL_PASS = "josj pnka spfg psao"
EMAIL_FROM = "jianzhou0420@gmail.com"
EMAIL_TO = ["jianzhou0420@outlook.com"]     # 可以同时发给多个人


def fetch_arxiv():
    """
    从 arXiv 获取最新搜索结果，返回 ID 列表和对应的条目字典
    每个条目包含：title, authors, categories, published, link
    """
    feed = feedparser.parse(ARXIV_API_URL)
    entries = {}
    for e in feed.entries:
        # arXiv ID 例如：arXiv:2101.00001v1
        aid = e.id.split("/")[-1]
        # e.tags 是一个列表，每个元素包含 'term' 字段，即分类名称
        cats = [t['term'] for t in e.tags] if hasattr(e, 'tags') else []
        entries[aid] = {
            "title": e.title.replace("\n", " ").strip(),
            "authors": [a.name for a in e.authors],
            "categories": cats,
            "published": e.published,
            "link": e.link
        }
    return entries


def load_previous():
    """
    从本地 JSON 文件加载上一次的结果
    """
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_current(entries):
    """
    将当前结果保存到本地 JSON 文件
    """
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def send_email(new_entries):
    """
    通过 SMTP 发送邮件通知新文章
    """
    lines = []
    for aid, info in new_entries.items():
        lines.append(f"- {info['title']}\n  Authors: {', '.join(info['authors'])}\n  Published: {info['published']}\n  Link: {info['link']}\n")

    body = "检测到 arXiv 上有新的 “manipulation” 相关文章：\n\n" + "\n".join(lines)
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = Header(EMAIL_FROM)
    msg["To"] = Header(", ".join(EMAIL_TO))
    msg["Subject"] = Header("【arXiv 更新通知】manipulation 相关新文章", "utf-8")

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())


def job():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始检测 arXiv 更新...")
    current = fetch_arxiv()
    previous = load_previous()

    # 按 current 的原始顺序，挑出 previous 中没有的 ID
    new_entries = {}
    for aid, info in current.items():
        if aid not in previous:
            new_entries[aid] = info

    if new_entries:
        print(f"发现 {len(new_entries)} 篇新文章，发送邮件通知。")
        send_email(new_entries)
    else:
        print("未发现新文章。")

    # 保存本次结果
    save_current(current)
    print("检测完成，结果已保存。\n")


if __name__ == "__main__":
    # 先执行一次，确保存在初始数据
    job()
    # 每小时执行一次
    schedule.every(1).hours.do(job)

    print("进入循环调度，每小时检测一次 arXiv 更新。")
    while True:
        schedule.run_pending()
        time.sleep(60)
