# graph/

Fetches Semantic Scholar data, clusters USC NLP group members by research interest, and outputs JSON for the website's D3.js graph.

## Setup

```bash
cd graph
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv/bin/python
```

## Usage

```bash
graph/.venv/bin/python graph/fetch_interests.py          # uses 3-month cache
graph/.venv/bin/python graph/fetch_interests.py --force   # re-fetch all
S2_API_KEY=key graph/.venv/bin/python graph/fetch_interests.py  # higher rate limits
```

## Adding people

1. Edit `data/people.yaml` — add name, role, lab, image, URL, and `semantic_scholar` profile URL
2. Run `fetch_interests.py` to regenerate `data/interests.json`

## Filtering bad papers

If S2 merges papers from a different author, add paper ID prefixes (first 12 hex chars) to `exclude_papers`:

```yaml
- name: Some Person
  semantic_scholar: https://...
  exclude_papers:
    - 51b118f6851a  # Wrong paper title
```

## Updating keywords

To improve topic extraction keywords (`STOPWORDS`, `COMPOUND_TERMS`, `FILTER_STOPWORDS`), ask Claude:

> Read `graph/data/.s2cache.json` and suggest keywords/compound terms to add based on paper titles.

## How it works

1. Fetches papers from S2 API per person (cached 3 months)
2. Extracts topic vectors from paper title keywords (weighted by recency + citations)
3. Builds co-authorship matrix (matches by S2 ID and name)
4. Clusters via spectral clustering on `0.4 * coauthorship + 0.6 * topic_similarity`
5. Auto-computes popular research area keywords for frontend filter pills
6. Outputs `data/interests.json`
