"""
graph_factory.py — Synthetic rumour propagation graph builder

Builds a directed graph of news articles, social posts, and rumour seeds
with planted disinformation origins. Designed so the ground truth is
fully deterministic (given a scenario config + RNG seed) and completely
hidden from the agent during the episode.

Graph structure:
  - Seed nodes:    1–3 planted disinformation "origins"
  - Article nodes: credible and non-credible news articles
  - Post nodes:    social media posts that share/paraphrase/quote/refute

Propagation model:
  Each non-credible article traces back to exactly one seed via a chain
  of edges. Credible articles are independent. Posts fan out from articles.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from models import Node, Edge, Claim


# ---------------------------------------------------------------------------
# Scenario templates (static text pools)
# ---------------------------------------------------------------------------

_DISINFO_TEMPLATES = [
    ("Scientists confirm 5G causes memory loss in birds",
     "A new study from the Institute of Wireless Biology has confirmed that 5G towers cause memory loss in migratory birds, disrupting their navigation.",
     0.91),
    ("Government secretly adding fluoride to boost obedience",
     "Leaked memos from health agencies suggest water fluoridation programmes are designed to suppress critical thinking among the population.",
     0.87),
    ("New vaccine linked to spontaneous magnetism",
     "Hundreds of recipients of the latest booster report that metal objects stick to the injection site, a phenomenon doctors are refusing to acknowledge.",
     0.93),
    ("Chemtrails confirmed to contain mind-altering compounds",
     "An independent lab analysis of rainwater collected near airports found trace psychoactive compounds consistent with aerosol spraying programmes.",
     0.89),
    ("Eating raw garlic daily eliminates 99% of viruses, study says",
     "Researchers at an Eastern European university claim that daily consumption of raw garlic activates antiviral genes previously dormant in humans.",
     0.72),
]

_REAL_TEMPLATES = [
    ("Council approves new cycling infrastructure budget",
     "The city council voted 7–2 to allocate £2.4 million for protected cycle lanes in the town centre, with construction expected next spring.",
     0.05),
    ("Local library extends opening hours for exam season",
     "From next Monday the central library will open until 10pm on weekdays to support students preparing for end-of-year examinations.",
     0.03),
    ("Rain forecast for the weekend, temperatures to drop",
     "Meteorologists expect heavy rain to move in from the west on Saturday, with temperatures falling to around 9°C by Sunday evening.",
     0.02),
    ("New café opens in converted Victorian post office",
     "A specialty coffee shop has opened its doors in the renovated 1887 post office building on Market Street, creating 12 local jobs.",
     0.04),
]

_POST_VERBS = [
    "Can't believe this! Sharing widely. 🚨",
    "This is exactly what they don't want you to know.",
    "Mainstream media won't touch this story.",
    "My aunt sent me this and now I'm scared.",
    "Scientists have known this for years.",
    "Please share before they delete it.",
    "Just a reminder that this is happening.",
    "Thread on why this matters ↓",
    "This explains everything.",
    "Do your research.",
]

_POST_REFUTALS = [
    "This is false. Here's the actual peer-reviewed source.",
    "Fact-check: this claim has been debunked by multiple labs.",
    "Please stop sharing this. It's misinformation.",
    "Context: the original study said the opposite of this.",
]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_scenario_graph(cfg: dict, rng: random.Random) -> dict:
    """
    Build nodes, edges, and claims from a scenario config.

    cfg keys:
      n_seeds        : int   — number of planted disinfo seeds (default 2)
      n_articles     : int   — total articles (default 10)
      n_posts        : int   — total posts (default 15)
      n_claims       : int   — standalone claims to score (default 4)
      disinfo_ratio  : float — fraction of articles that are disinfo (default 0.4)
    """
    n_seeds       = cfg.get("n_seeds", 2)
    n_articles    = cfg.get("n_articles", 10)
    n_posts       = cfg.get("n_posts", 15)
    n_claims      = cfg.get("n_claims", 4)
    disinfo_ratio = cfg.get("disinfo_ratio", 0.4)

    base_time = datetime(2024, 3, 15, 8, 0, 0)
    nodes: list[Node] = []
    edges: list[Edge] = []
    _id  = _counter()

    # --- Seed nodes (planted origins of disinfo) -------------------------
    seed_ids = []
    disinfo_pool = rng.sample(_DISINFO_TEMPLATES, min(n_seeds, len(_DISINFO_TEMPLATES)))
    for i, (title, text, sev) in enumerate(disinfo_pool):
        nid = f"seed_{next(_id)}"
        ts  = (base_time - timedelta(hours=rng.randint(6, 24))).isoformat()
        nodes.append(Node(
            node_id=nid, kind="seed", text=text,
            timestamp=ts, source=f"anon_forum_{i}",
            credibility=0.05,
            share_count=rng.randint(1000, 8000),
            _true_label="disinformation", _is_origin=True,
        ))
        seed_ids.append(nid)

    # --- Article nodes ---------------------------------------------------
    n_disinfo_articles = int(n_articles * disinfo_ratio)
    n_real_articles    = n_articles - n_disinfo_articles

    disinfo_extras = list(_DISINFO_TEMPLATES)
    rng.shuffle(disinfo_extras)
    real_pool = list(_REAL_TEMPLATES)
    rng.shuffle(real_pool)

    article_ids:      list[str] = []
    disinfo_article_ids: list[str] = []

    for i in range(n_disinfo_articles):
        template = disinfo_extras[i % len(disinfo_extras)]
        title, text, _ = template
        nid = f"art_{next(_id)}"
        ts  = (base_time + timedelta(hours=rng.randint(1, 6))).isoformat()
        nodes.append(Node(
            node_id=nid, kind="article",
            text=f"BREAKING: {title}. {text}",
            timestamp=ts,
            source=rng.choice(["globalfreemind.net", "truthwatcher.io", "realfacts24.com"]),
            credibility=rng.uniform(0.05, 0.25),
            share_count=rng.randint(500, 5000),
            _true_label="disinformation",
        ))
        article_ids.append(nid)
        disinfo_article_ids.append(nid)
        # Connect to a seed
        seed = rng.choice(seed_ids)
        edges.append(Edge(src=seed, dst=nid, relation="shares"))

    for i in range(n_real_articles):
        t = real_pool[i % len(real_pool)]
        nid = f"art_{next(_id)}"
        ts  = (base_time + timedelta(hours=rng.randint(0, 8))).isoformat()
        nodes.append(Node(
            node_id=nid, kind="article",
            text=t[1],
            timestamp=ts,
            source=rng.choice(["localgazette.co.uk", "cityreporter.net", "eveningherald.com"]),
            credibility=rng.uniform(0.75, 0.98),
            share_count=rng.randint(20, 400),
            _true_label="real",
        ))
        article_ids.append(nid)

    rng.shuffle(article_ids)

    # --- Post nodes ------------------------------------------------------
    for _ in range(n_posts):
        nid = f"post_{next(_id)}"
        ts  = (base_time + timedelta(hours=rng.randint(2, 18))).isoformat()
        src_article = rng.choice(article_ids)
        src_node    = next(n for n in nodes if n.node_id == src_article)
        is_disinfo  = src_node._true_label == "disinformation"

        if is_disinfo:
            verb   = rng.choice(_POST_VERBS)
            text   = f"{verb} — {src_node.text[:120]}..."
            label  = rng.choice(["disinformation", "misleading"])
            cred   = rng.uniform(0.05, 0.3)
            shares = rng.randint(100, 3000)
            rel    = rng.choice(["shares", "paraphrases", "quotes"])
        else:
            if rng.random() < 0.2:
                text  = rng.choice(_POST_REFUTALS) + f" (re: {src_node.text[:80]}...)"
                label = "real"
                rel   = "refutes"
            else:
                text  = f"Interesting local news: {src_node.text[:120]}..."
                label = "real"
                rel   = "shares"
            cred   = rng.uniform(0.5, 0.85)
            shares = rng.randint(5, 150)

        nodes.append(Node(
            node_id=nid, kind="post",
            text=text, timestamp=ts,
            source=f"@user_{rng.randint(1000, 9999)}",
            credibility=cred,
            share_count=shares,
            _true_label=label,
        ))
        edges.append(Edge(src=src_article, dst=nid, relation=rel))

    # --- Add some cross-article propagation edges (disinfo re-shares) ---
    for _ in range(max(1, len(disinfo_article_ids) // 2)):
        if len(disinfo_article_ids) >= 2:
            a, b = rng.sample(disinfo_article_ids, 2)
            edges.append(Edge(src=a, dst=b, relation="paraphrases"))

    # --- Claims ----------------------------------------------------------
    claims: list[Claim] = []
    disinfo_claims = rng.sample(disinfo_pool, min(n_claims, len(disinfo_pool)))
    for i, (_, text, sev) in enumerate(disinfo_claims):
        claims.append(Claim(
            claim_id=f"claim_{i}",
            text=text,
            _true_severity=sev,
        ))

    return {"nodes": nodes, "edges": edges, "claims": claims}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counter():
    i = 0
    while True:
        yield i
        i += 1
