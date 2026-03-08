#!/usr/bin/env python3
"""
Fetch research interests from Semantic Scholar and cluster USC NLP group members.

Reads people from graph/data/people.yaml (with semantic_scholar URLs).
Caches API responses in graph/data/.s2cache.json — skips anyone fetched
within the last 3 months.

Usage:
    python fetch_interests.py              # normal run (skip recent)
    python fetch_interests.py --force      # re-fetch everyone

Optional: set S2_API_KEY env var for higher rate limits.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import yaml
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# ── Config ──
API_SLEEP = 3.0
MAX_RETRIES = 3
CACHE_DAYS = 90  # skip if fetched within this many days

S2_BASE = "https://api.semanticscholar.org/graph/v1"
GRAPH_DIR = Path(__file__).resolve().parent
DATA_DIR = GRAPH_DIR / "data"
PEOPLE_YAML = DATA_DIR / "people.yaml"
CACHE_JSON = DATA_DIR / ".s2cache.json"
OUTPUT_JSON = DATA_DIR / "interests.json"

CURRENT_YEAR = datetime.now().year
RECENCY_BOOST_YEARS = 3



# ── Helpers ──

def filter_papers(name: str, details: dict, excluded: set[str] | None = None) -> dict:
    """Return a copy of details with excluded papers removed (does not mutate original)."""
    if not excluded or not details:
        return details
    papers = details.get("papers") or []
    filtered = [p for p in papers if p.get("paperId", "")[:12] not in excluded]
    if len(filtered) < len(papers):
        print(f"    Filtered {len(papers) - len(filtered)} excluded papers for {name}")
        details = {**details, "papers": filtered}
    return details


def extract_s2id(url: str) -> str | None:
    if not url:
        return None
    match = re.search(r'/(\d+)/?$', url.strip().rstrip('/'))
    return match.group(1) if match else None


def load_people() -> list[dict]:
    with open(PEOPLE_YAML) as f:
        people = yaml.safe_load(f)
    for p in people:
        s2_url = p.get("semantic_scholar") or ""
        p["s2id"] = extract_s2id(str(s2_url)) if s2_url else None
        raw = p.get("exclude_papers") or []
        p["exclude_papers"] = set(str(pid).strip() for pid in raw) if raw else None
    return people


def load_cache() -> dict:
    if CACHE_JSON.exists():
        with open(CACHE_JSON) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_JSON.parent.mkdir(exist_ok=True)
    with open(CACHE_JSON, "w") as f:
        json.dump(cache, f, indent=2)


def is_cache_fresh(entry: dict) -> bool:
    fetched = entry.get("fetched_date")
    if not fetched:
        return False
    fetched_dt = datetime.fromisoformat(fetched)
    return datetime.now() - fetched_dt < timedelta(days=CACHE_DAYS)


# ── API ──

def make_session() -> requests.Session:
    session = requests.Session()
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        session.headers["x-api-key"] = api_key
        print(f"Using S2 API key: {api_key[:8]}...")
    else:
        print("No S2_API_KEY set — using unauthenticated access (lower rate limits)")
    return session


def api_request(session, url, params, timeout=15):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                time.sleep((attempt + 1) * 10)
                continue
            print(f"    HTTP error: {e}")
            return None
        except Exception as e:
            print(f"    Error: {e}")
            return None
    print(f"    Failed after {MAX_RETRIES} retries")
    return None


def fetch_author_details(author_id: str, session: requests.Session) -> dict | None:
    fields = "name,affiliations,paperCount,papers.title,papers.year,papers.venue,papers.fieldsOfStudy,papers.s2FieldsOfStudy,papers.authors,papers.citationCount"
    return api_request(
        session, f"{S2_BASE}/author/{author_id}",
        params={"fields": fields}, timeout=30,
    )


# ── Topic extraction from paper titles ──

STOPWORDS = frozenset(
    "a an the and or but in on of for to with by from is are was were be been "
    "being at as it its this that these those do does did not no nor so if how "
    "what which who whom where when why can could will would shall should may "
    "might must about above after again against all am any both each few has "
    "have having he her here hers herself him himself his i into me more most "
    "my myself off once only other our ours ourselves out over own same she "
    "some such than them then there they their theirs themselves through too "
    "under until up us very we you your yours yourself yourselves also between "
    "during just because while using via beyond towards toward among within upon "
    "without along across before after another however still yet much new well "
    "based approach approaches method methods model models framework task tasks "
    "paper study work use used data set dataset datasets results show shows "
    "shown analysis experiments experiment experimental case cases evidence "
    "towards toward novel effective efficient simple better improved improving "
    "learning explore exploring large scale small end".split()
)

# Compound terms to detect as single keywords
COMPOUND_TERMS = {
    ("language", "model"): "language models",
    ("language", "models"): "language models",
    ("large", "language"): "large language models",
    ("machine", "learning"): "machine learning",
    ("deep", "learning"): "deep learning",
    ("reinforcement", "learning"): "reinforcement learning",
    ("transfer", "learning"): "transfer learning",
    ("natural", "language"): "natural language processing",
    ("knowledge", "graph"): "knowledge graphs",
    ("knowledge", "graphs"): "knowledge graphs",
    ("question", "answering"): "question answering",
    ("text", "generation"): "text generation",
    ("machine", "translation"): "machine translation",
    ("sentiment", "analysis"): "sentiment analysis",
    ("named", "entity"): "named entity recognition",
    ("information", "extraction"): "information extraction",
    ("information", "retrieval"): "information retrieval",
    ("common", "sense"): "commonsense reasoning",
    ("commonsense", "reasoning"): "commonsense reasoning",
    ("visual", "question"): "visual question answering",
    ("multi", "modal"): "multimodal",
    ("cross", "lingual"): "cross-lingual",
    ("zero", "shot"): "zero-shot",
    ("few", "shot"): "few-shot",
    ("in", "context"): "in-context learning",
    ("pre", "training"): "pre-training",
    ("pre", "trained"): "pre-training",
    ("fine", "tuning"): "fine-tuning",
    ("fine", "tune"): "fine-tuning",
    ("text", "classification"): "text classification",
    ("relation", "extraction"): "relation extraction",
    ("dialogue", "systems"): "dialogue systems",
    ("dialogue", "system"): "dialogue systems",
    ("robot", "learning"): "robot learning",
    ("code", "generation"): "code generation",
    ("instruction", "tuning"): "instruction tuning",
    ("instruction", "following"): "instruction tuning",
    ("chain", "thought"): "chain-of-thought",
    ("reward", "model"): "reward models",
    ("data", "augmentation"): "data augmentation",
    ("event", "extraction"): "event extraction",
    ("knowledge", "distillation"): "knowledge distillation",
    ("language", "understanding"): "language understanding",
    ("language", "generation"): "language generation",
    ("image", "generation"): "image generation",
    ("object", "detection"): "object detection",
    ("speech", "recognition"): "speech recognition",
    ("neural", "network"): "neural networks",
    ("neural", "networks"): "neural networks",
    ("graph", "neural"): "graph neural networks",
    ("attention", "mechanism"): "attention",
    ("transformer", "models"): "transformers",
    ("safety", "alignment"): "AI safety",
    ("human", "evaluation"): "human evaluation",
    ("prompt", "tuning"): "prompt engineering",
    ("prompt", "engineering"): "prompt engineering",
}


def tokenize_title(title: str) -> list[str]:
    """Lowercase, remove punctuation, split into words."""
    title = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return [w for w in title.split() if len(w) > 1]


def extract_keywords_from_title(words: list[str]) -> list[str]:
    """Extract unigrams and compound terms from a tokenized title."""
    keywords = []
    i = 0
    while i < len(words):
        # Try bigram compound match
        if i + 1 < len(words):
            bigram = (words[i], words[i + 1])
            compound = COMPOUND_TERMS.get(bigram)
            if compound:
                keywords.append(compound)
                i += 2
                continue
        # Single word (skip stopwords and short tokens)
        if words[i] not in STOPWORDS and len(words[i]) > 2:
            keywords.append(words[i])
        i += 1
    return keywords


def extract_topics(details: dict) -> dict[str, float]:
    """Extract keyword-based topics from an author's paper titles."""
    topic_scores: dict[str, float] = {}
    for paper in (details.get("papers") or []):
        title = paper.get("title") or ""
        if not title:
            continue
        year = paper.get("year") or 0
        citations = paper.get("citationCount") or 0
        recency_weight = 2.0 if (CURRENT_YEAR - year) <= RECENCY_BOOST_YEARS else 1.0
        citation_weight = 1.0 + np.log1p(citations) * 0.1
        weight = recency_weight * citation_weight

        words = tokenize_title(title)
        keywords = extract_keywords_from_title(words)
        for kw in keywords:
            topic_scores[kw] = topic_scores.get(kw, 0) + weight

    return topic_scores


def build_topic_vectors(all_topics):
    topic_set = set()
    for topics in all_topics.values():
        topic_set.update(topics.keys())
    topic_list = sorted(topic_set)
    topic_idx = {t: i for i, t in enumerate(topic_list)}

    names = sorted(all_topics.keys())
    matrix = np.zeros((len(names), len(topic_list)))
    for i, name in enumerate(names):
        for topic, score in all_topics[name].items():
            matrix[i, topic_idx[topic]] = score

    return names, matrix, topic_list


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-ish, collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", name.lower().strip())


def build_coauthorship_matrix(people_names, author_details):
    n = len(people_names)
    name_idx = {name: i for i, name in enumerate(people_names)}
    matrix = np.zeros((n, n))

    # Match by S2 author ID
    s2_id_to_name = {}
    for name, details in author_details.items():
        if details:
            s2_id_to_name[details.get("authorId")] = name

    # Also match by normalized name (handles S2 fragmenting one person into multiple IDs)
    norm_to_name = {}
    for name in people_names:
        norm_to_name[_normalize_name(name)] = name

    for name, details in author_details.items():
        if not details or name not in name_idx:
            continue
        i = name_idx[name]
        for paper in (details.get("papers") or []):
            for a in (paper.get("authors") or []):
                # Try ID match first, then name match
                coauthor = s2_id_to_name.get(a.get("authorId"))
                if not coauthor:
                    coauthor = norm_to_name.get(_normalize_name(a.get("name", "")))
                if coauthor and coauthor != name and coauthor in name_idx:
                    matrix[i, name_idx[coauthor]] += 1

    return np.maximum(matrix, matrix.T)


def cluster_people(names, topic_matrix, coauth_matrix, alpha=0.4):
    n = len(names)
    if n < 4:
        return np.zeros(n, dtype=int), 1

    norms = np.linalg.norm(topic_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = topic_matrix / norms
    topic_sim = normalized @ normalized.T
    np.fill_diagonal(topic_sim, 0)

    coauth_max = coauth_matrix.max()
    coauth_norm = np.log1p(coauth_matrix) / np.log1p(coauth_max) if coauth_max > 0 else coauth_matrix

    affinity = alpha * coauth_norm + (1 - alpha) * topic_sim
    np.fill_diagonal(affinity, 0)
    affinity = np.maximum(affinity, 0)

    best_k, best_score, best_labels = 4, -1, None
    for k in range(3, min(9, n)):
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42, n_init=10)
            labels = sc.fit_predict(affinity + np.eye(n))
            score = silhouette_score(affinity, labels, metric="precomputed")
            print(f"  k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
        except Exception as e:
            print(f"  k={k}: failed: {e}")

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    print(f"  Selected k={best_k} (silhouette={best_score:.3f})")
    return best_labels, best_k


def generate_cluster_labels(names, labels, all_topics):
    clusters = {}
    for name, label in zip(names, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = {"id": label, "members": [], "topic_scores": {}}
        clusters[label]["members"].append(name)
        for topic, score in all_topics.get(name, {}).items():
            clusters[label]["topic_scores"][topic] = clusters[label]["topic_scores"].get(topic, 0) + score

    result = []
    for cid, info in sorted(clusters.items()):
        sorted_topics = sorted(info["topic_scores"].items(), key=lambda x: -x[1])
        top_topics = [t for t, _ in sorted_topics[:3]]
        label_str = " & ".join(top_topics[:2]) if top_topics else f"Cluster {cid}"
        result.append({"id": cid, "label": label_str, "topics": top_topics})
    return result


# Minimum number of people a keyword must appear in (top-N topics) to be a filter tag
MIN_KEYWORD_PEOPLE = 4
MAX_FILTER_KEYWORDS = 12

# Generic words that are too vague to be useful as filter labels
FILTER_STOPWORDS = frozenset(
    "multi language learn learning model models neural network networks "
    "text self word words based guided modeling inference action features "
    "representation representations level order online training "
    "llm llms visual evaluating".split()
)


def compute_popular_keywords(all_topics: dict[str, dict[str, float]], top_n: int = 10) -> list[str]:
    """Return the most popular keywords across all people (by how many people have it in top-N)."""
    from collections import Counter
    people_count = Counter()
    for person_topics in all_topics.values():
        top_keys = set(t for t, _ in sorted(person_topics.items(), key=lambda x: -x[1])[:top_n])
        for k in top_keys:
            if k not in FILTER_STOPWORDS:
                people_count[k] += 1
    popular = [(kw, cnt) for kw, cnt in people_count.most_common() if cnt >= MIN_KEYWORD_PEOPLE]
    return [kw for kw, _ in popular[:MAX_FILTER_KEYWORDS]]


def make_id(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build_output(people, author_details, all_topics, names_ordered, topic_matrix, coauth_matrix, labels, cluster_info, cache):
    name_idx = {name: i for i, name in enumerate(names_ordered)}

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
    for ci in cluster_info:
        ci["color"] = colors[ci["id"] % len(colors)]

    # Compute popular keywords for filter tags
    popular_keywords = compute_popular_keywords(all_topics)
    popular_set = set(popular_keywords)

    nodes = []
    for person in people:
        name = person["name"]
        idx = name_idx.get(name)
        details = author_details.get(name)
        person_topics = all_topics.get(name, {})
        sorted_topics = sorted(person_topics.items(), key=lambda x: -x[1])

        # Tag person with popular keywords that appear in their top-10 topics
        top_keys = set(t for t, _ in sorted_topics[:10])
        person_tags = [kw for kw in popular_keywords if kw in top_keys]

        top_papers = []
        if details:
            papers = sorted(details.get("papers") or [], key=lambda p: -(p.get("year") or 0))
            for p in papers[:5]:
                top_papers.append({
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "venue": p.get("venue", ""),
                    "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                    "citations": p.get("citationCount", 0),
                })

        cache_entry = cache.get(name, {})
        nodes.append({
            "id": make_id(name),
            "name": name,
            "role": person["role"],
            "lab": person.get("lab", ""),
            "url": person.get("url", ""),
            "image": person.get("image", ""),
            "cluster": int(labels[idx]) if idx is not None else 0,
            "topics": [t for t, _ in sorted_topics[:5]],
            "researchAreas": person_tags,
            "paperCount": details.get("paperCount", 0) if details else 0,
            "topPapers": top_papers,
            "semanticScholarUrl": cache_entry.get("semantic_scholar_url", ""),
            "fetchedDate": cache_entry.get("fetched_date", ""),
        })

    # Edges
    edges = []
    n = len(names_ordered)
    norms = np.linalg.norm(topic_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = topic_matrix / norms
    topic_sim = normalized @ normalized.T
    coauth_max = coauth_matrix.max() if coauth_matrix.max() > 0 else 1

    for i in range(n):
        for j in range(i + 1, n):
            coauth_count = int(coauth_matrix[i, j])
            t_sim = float(topic_sim[i, j])
            coauth_n = float(np.log1p(coauth_matrix[i, j]) / np.log1p(coauth_max))
            weight = 0.4 * coauth_n + 0.6 * t_sim
            if weight > 0.15 or coauth_count > 0:
                edges.append({
                    "source": make_id(names_ordered[i]),
                    "target": make_id(names_ordered[j]),
                    "coauthorCount": coauth_count,
                    "topicSimilarity": round(t_sim, 3),
                    "weight": round(weight, 3),
                })

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "researchAreas": popular_keywords,
        "nodes": nodes,
        "edges": edges,
        "clusters": cluster_info,
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch all, ignoring cache")
    args = parser.parse_args()

    print("=== USC NLP Research Interest Clustering ===\n")

    # 1. Load people
    people = load_people()
    with_s2 = [p for p in people if p.get("s2id")]
    without_s2 = [p for p in people if not p.get("s2id")]
    print(f"{len(people)} people, {len(with_s2)} with S2 URLs, {len(without_s2)} without")

    if not with_s2:
        print("\nNo semantic_scholar URLs found in graph/data/people.yaml!")
        return

    if without_s2:
        print(f"  Missing: {', '.join(p['name'] for p in without_s2)}")
        print("  (will be assigned to their lab faculty's cluster)\n")

    # 2. Load cache, fetch as needed
    cache = load_cache()
    session = make_session()
    author_details = {}
    fetched_count = 0
    skipped_count = 0

    print(f"\nFetching author details...")
    for person in with_s2:
        name = person["name"]
        s2id = person["s2id"]
        s2_url = person.get("semantic_scholar", "")
        excluded = person.get("exclude_papers")
        cached = cache.get(name)

        if cached and is_cache_fresh(cached) and not args.force:
            print(f"  {name} — skipped (fetched {cached['fetched_date'][:10]})")
            author_details[name] = filter_papers(name, cached["details"], excluded)
            skipped_count += 1
            continue

        print(f"  {name} ({s2id}) — fetching...")
        details = fetch_author_details(s2id, session)
        if details:
            author_details[name] = filter_papers(name, details, excluded)
            cache[name] = {
                "semantic_scholar_url": s2_url,
                "fetched_date": datetime.now().isoformat(timespec="seconds"),
                "details": details,
            }
            fetched_count += 1
        else:
            print(f"    WARNING: could not fetch {name}")
            if cached:
                print(f"    Using stale cache from {cached['fetched_date'][:10]}")
                author_details[name] = filter_papers(name, cached["details"], excluded)
        time.sleep(API_SLEEP)

    save_cache(cache)
    print(f"\n  Fetched {fetched_count}, skipped {skipped_count} (cached), {len(author_details)} total\n")

    # 3. Extract topics
    print("Extracting topics...")
    all_topics = {}
    for name, details in author_details.items():
        topics = extract_topics(details)
        all_topics[name] = topics
        top3 = sorted(topics.items(), key=lambda x: -x[1])[:3]
        print(f"  {name}: {', '.join(t for t, _ in top3)}")

    # 4. Build matrices
    print("\nBuilding matrices...")
    names_ordered, topic_matrix, topic_names = build_topic_vectors(all_topics)
    coauth_matrix = build_coauthorship_matrix(names_ordered, author_details)
    print(f"  {len(names_ordered)} people, {len(topic_names)} topics, {int((coauth_matrix > 0).sum() / 2)} co-author pairs")

    # 5. Cluster
    print("\nClustering...")
    labels, best_k = cluster_people(names_ordered, topic_matrix, coauth_matrix)

    # 6. Assign people without S2 to their lab faculty's cluster
    cluster_info = generate_cluster_labels(names_ordered, labels, all_topics)
    people_in_matrix = set(names_ordered)

    faculty_clusters = {}
    for p in people:
        if p["role"] == "faculty" and p["name"] in people_in_matrix:
            idx = names_ordered.index(p["name"])
            faculty_clusters[p.get("lab", "")] = int(labels[idx])

    for p in without_s2:
        if p["name"] not in people_in_matrix:
            fallback = faculty_clusters.get(p.get("lab", ""), 0)
            names_ordered.append(p["name"])
            labels = np.append(labels, fallback)
            all_topics[p["name"]] = {}
            print(f"  {p['name']} (no S2) -> cluster {fallback}")

    cluster_info = generate_cluster_labels(names_ordered, labels, all_topics)
    for ci in cluster_info:
        members = [n for n, l in zip(names_ordered, labels) if l == ci["id"]]
        print(f"  Cluster {ci['id']} ({ci['label']}): {', '.join(members)}")

    # Expand matrices for added people
    n_orig = topic_matrix.shape[0]
    n_total = len(names_ordered)
    if n_total > n_orig:
        extra = n_total - n_orig
        topic_matrix = np.vstack([topic_matrix, np.zeros((extra, topic_matrix.shape[1]))])
        new_coauth = np.zeros((n_total, n_total))
        new_coauth[:n_orig, :n_orig] = coauth_matrix
        coauth_matrix = new_coauth

    # 7. Build & write output
    print("\nWriting output...")
    output = build_output(people, author_details, all_topics, names_ordered, topic_matrix, coauth_matrix, labels, cluster_info, cache)

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done! {OUTPUT_JSON}")
    print(f"  {len(output['nodes'])} nodes, {len(output['edges'])} edges, {len(output['clusters'])} clusters")


if __name__ == "__main__":
    main()
