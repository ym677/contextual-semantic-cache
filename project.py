"""Context-scoped semantic response cache with real SQLite persistence."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SPEC = {
    "title": "ContextCache · 语义缓存实验室",
    "subtitle": "让相近问题复用回答，同时隔离文档版本、模型与演示模式",
    "slug": "contextual-semantic-cache",
    "accent": "#0f766e",
    "input_label": "输入“上下文版本 || 问题”，版本相同才允许复用",
    "sample_input": "maintenance-v1 || 轴承温度超过阈值时应该如何处理？",
}

SEEDS = [
    ("轴承温度超过阈值时应该如何处理？", "本地样例回答：先核对传感器、阈值与运行工况，再依据对应设备维护手册处理。"),
    ("如何解释设备振动趋势？", "本地样例回答：比较同一设备的历史基线、运行工况和变化持续时间。"),
    ("为什么要记录设备维护日志？", "本地样例回答：维护日志用于追踪故障、处理过程和复检记录。"),
]


def normalized_query(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def guard_signature(text):
    text = normalized_query(text)
    numbers = tuple(re.findall(r"\d+(?:\.\d+)?", text))
    negations = tuple(re.findall(r"不|禁止|无|\b(?:not|never|without|no)\b", text))
    identifiers = tuple(re.findall(r"(?:设备|传感器|device|sensor)\s*([a-z0-9_-]+)", text))
    return numbers, negations, identifiers


class SemanticCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS entries (
                scope TEXT NOT NULL, query TEXT NOT NULL, original TEXT NOT NULL,
                answer TEXT NOT NULL, expires REAL NOT NULL,
                PRIMARY KEY(scope, query))""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def put(self, scope, query, answer, ttl=3600, now=None):
        if ttl <= 0:
            raise ValueError("缓存 TTL 必须大于 0。")
        if not query.strip() or not answer.strip():
            raise ValueError("缓存问题和回答不能为空。")
        expires = (time.time() if now is None else now) + ttl
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?)",
                (scope, normalized_query(query), query, answer, expires),
            )

    def lookup(self, scope, query, threshold=0.88, now=None):
        if not query.strip() or not 0 <= threshold <= 1:
            raise ValueError("请输入问题，并将相似度阈值设为 0～1。")
        clock = time.time() if now is None else now
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT query, original, answer FROM entries "
                "WHERE scope = ? AND expires > ? ORDER BY expires DESC LIMIT 500",
                (scope, clock),
            ).fetchall()
        query_key = normalized_query(query)
        for key, original, answer in rows:
            if key == query_key:
                return {"kind": "exact", "answer": answer, "score": 1.0,
                        "candidates": [{"query": original, "score": 1.0}]}
        if not rows:
            return {"kind": "miss", "answer": None, "score": 0.0, "candidates": []}
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        matrix = vectorizer.fit_transform([row[0] for row in rows] + [query_key])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        order = scores.argsort()[::-1]
        signature = guard_signature(query)
        eligible = [int(index) for index in order if guard_signature(rows[int(index)][1]) == signature]
        best = eligible[0] if eligible else int(order[0])
        candidates = [{"query": rows[int(index)][1], "score": round(float(scores[index]), 4),
                       "guard": guard_signature(rows[int(index)][1]) == signature}
                      for index in order[:5]]
        hit = bool(eligible) and float(scores[best]) >= threshold
        return {"kind": "semantic" if hit else "miss", "score": float(scores[best]),
                "answer": rows[best][2] if hit else None, "candidates": candidates}


def make_scope(context, mode, settings):
    # Do not include credentials or other private settings in the database.
    identity = {"context": context, "mode": mode,
                "base_url": settings.get("LLM_BASE_URL", ""),
                "model": settings.get("LLM_MODEL", "local-template"),
                "policy": "maintenance-cautious-v1"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def run(payload, client):
    text = payload["text"].strip()
    context, separator, query = text.partition("||")
    if not separator:
        context, query = "maintenance-v1", text
    context, query = context.strip(), query.strip()
    if not context or not query:
        raise ValueError("上下文版本和问题都需要填写。")
    mode = "live" if client.enabled else "demo"
    scope = make_scope(context, mode, client.settings)
    cache = SemanticCache(Path(__file__).parent / ".cache" / "responses.sqlite3")
    if not client.enabled and context == "maintenance-v1":
        # Seed entries are clearly labelled authored examples, never cloud answers.
        for question, answer in SEEDS:
            if cache.lookup(scope, question)["kind"] == "miss":
                cache.put(scope, question, answer)
    result = cache.lookup(scope, query)
    trace = [f"上下文 {context}", f"隔离空间 {scope[:12]}", f"查询路径 {result['kind']}"]
    if result["answer"] is None:
        if client.enabled:
            answer = client.complete(
                "你是设备维护知识学习助手。避免虚构设备规格。资料不足时说明需要手册和运行工况。",
                f"上下文版本：{context}\n问题：{query}", max_tokens=400,
            )
            trace.append("缓存未命中，实际调用云端 API 后写入缓存")
        else:
            answer = f"本地模板演示：已记录问题“{query}”。真实回答需切换云端模式。"
            trace.append("缓存未命中，写入本地模板演示结果，未调用模型")
        cache.put(scope, query, answer)
    else:
        answer = result["answer"]
        trace.append("缓存命中，本次不调用模型")
    return {
        "summary": answer,
        "metrics": {"缓存路径": result["kind"], "最高相似度": round(result["score"], 4),
                    "复用阈值": 0.88, "TTL 秒": 3600},
        "sections": [
            {"title": "请求轨迹", "items": trace},
            {"title": "候选缓存", "table": {"columns": ["历史问题", "余弦相似度", "数字/否定/设备编号校验"],
                "rows": [[item["query"], item["score"], "通过" if item.get("guard", True) else "拒绝复用"] for item in result["candidates"]]}},
            {"title": "实验边界", "text": "字符 TF-IDF 衡量表面相似性，不保证语义等价。含数字、否定词或业务条件的变化应人工核对。演示与云端、不同模型和上下文版本使用独立空间。"},
        ],
    }
