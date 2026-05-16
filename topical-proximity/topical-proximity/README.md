# topical-proximity

LLM-based content audit classifier. Replaces cosine similarity with multi-dimensional classification so you can actually tell a buyer's guide from a product review.

Works with **OpenAI** or **Anthropic (Claude)** — switch with a single config flag.

## The problem

Cosine similarity puts "solar installation cost" and "Tesla Powerwall 3 review" in the same neighborhood. Both are about solar. The words match.

But one is a hire-me page and the other is mid-funnel product research. Pruning or consolidating based on that similarity destroys the site.

Embeddings tell you what a post is *about*. They can't tell you what it *is*.

## What this does

For each blog post on a site, it produces:

| Column | Meaning |
|---|---|
| `service` | Which of the business's services this post supports, or `NONE` |
| `service_alignment` | 0-1 score for how directly it supports that service |
| `post_type` | `SERVICE_INTENT`, `PRODUCT_REVIEW`, `COMPARISON`, `INSTALLATION_GUIDE`, `EDUCATIONAL`, `DESIGN_INSPIRATION`, `DRIFT`, `OFF_TOPIC` |
| `commercial_intent` | `HIGH` / `MEDIUM` / `LOW` |
| `tier` | Final pruning bucket: `CORE_SERVICE`, `CORE_PRODUCT`, `ADJACENT_COMPARISON`, `ADJACENT_INFORMATIONAL`, `PERIPHERAL`, `OFF_TOPIC` |
| `reasoning` | One paragraph explaining why |
| `overridden` | `True` if the verification pass corrected the first-pass classification |

The tier column is what you sort by for prune-or-keep decisions. Everything else is the audit trail.

## How it's different from cosine-based audits

1. **LLM classification, not embedding similarity.** The model reasons about what each post is before it classifies. A "Tesla Powerwall review" gets tagged `PRODUCT_REVIEW`, not bucketed with hire-me content like "Powerwall installation cost."
2. **Reads actual page content, not just titles.** If your CSV has a URL column and no `Content`, the script fetches and extracts the body. Title alone is too lossy — "SunPower Maxeon Panels Review" could be a 3000-word technical breakdown or a 400-word affiliate post.
3. **Two-pass verification on ambiguous rows.** Anything scoring between 0.35 and 0.65 on service alignment gets independently re-classified with override capability. Catches the silent errors single-pass classification misses.
4. **Few-shot anchored.** Six hand-picked edge cases live in the prompt to nail the boundaries where models drift (PRODUCT_REVIEW vs SERVICE_INTENT, DRIFT vs OFF_TOPIC).
5. **Provider-agnostic.** Same schema, same outputs, same retry logic — works with OpenAI or Anthropic. Switch in one line.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/topical-proximity.git
cd topical-proximity
pip install -e .
```

Then set the API key for whichever provider you're using:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
topical-proximity \
  --config examples/config.yaml \
  --services examples/services.csv \
  --audit examples/audit.csv \
  --output topical_proximity.csv
```

Or import as a library:

```python
from topical_proximity import run

run(
    config_path="config.yaml",
    services_path="services.csv",
    audit_path="audit.csv",
    output_path="topical_proximity.csv",
)
```

## Configure

Edit `config.yaml`:

```yaml
business:
  name: "Helios Solar"
  type: "residential solar installation company"
  location: "Austin, TX"
  description: |
    Helios Solar is a residential solar installer serving the Austin metro.
    Services include solar panel installation, battery backup systems, EV
    charger installation, and ongoing maintenance.

classification:
  provider: "openai"        # or "anthropic"
  model: "gpt-5.4"          # see model table below
  workers: 8
  fetch_content: true
  content_max_chars: 4000
  verify_alignment_range: [0.35, 0.65]
  checkpoint_every: 25
```

### Choosing a provider and model

| Provider | Model | When to use |
|---|---|---|
| `openai` | `gpt-5.4` | Default. Strong classification, fast. |
| `anthropic` | `claude-opus-4-7` | Best quality. Use when accuracy matters more than cost. |
| `anthropic` | `claude-sonnet-4-6` | Balanced. Most users should start here on the Claude side. |
| `anthropic` | `claude-haiku-4-5` | Cheapest. Good for first passes on huge datasets. |

To switch from OpenAI to Claude, change two lines:

```yaml
classification:
  provider: "anthropic"
  model: "claude-opus-4-7"
```

That's it. Same schema, same outputs, same verification pass.

### Writing a good `description`

This field gets injected into the LLM prompt — write it like you're briefing a strategist, not like marketing copy. What does the business actually do? What do they NOT do? That context drives the OFF_TOPIC / DRIFT calls. A solar installer doesn't sell backyard solar lights — describing that explicitly is what stops "Best Backyard Solar Lights of 2025" from getting filed as CORE_PRODUCT.

## Input format

**`services.csv`** — one row per service:

```csv
Service_Name,Descriptor
Residential Solar Installation,"Full residential solar panel system design and installation. Includes site assessment, permitting, and grid interconnection."
Battery Backup Systems,"Home battery storage installation including Tesla Powerwall, Enphase IQ Battery, and FranklinWH."
```

**`audit.csv`** — one row per post. Required: `URL`. Recommended: `Title`, `H1`, `Meta Description`. Optional: `Topic`, `Content` (if you already have body text scraped, paste it here and skip the fetch step).

```csv
URL,Title,H1,Meta Description
https://example.com/blog/solar-installation-cost,"How Much Does Solar Installation Cost in 2025?","Solar Installation Cost Guide","Detailed cost breakdown..."
```

## Cost & runtime

Rough estimates for 500 posts including the verification pass:

| Setup | Cost | Runtime |
|---|---|---|
| OpenAI `gpt-5.4` | ~$3-5 | 5-15 min |
| Anthropic `claude-opus-4-7` | ~$8-12 | 10-20 min |
| Anthropic `claude-sonnet-4-6` | ~$3-5 | 5-15 min |
| Anthropic `claude-haiku-4-5` | ~$1-2 | 5-10 min |

Fetching content adds a couple of minutes depending on target site latency.

If you're cost-conscious, set `fetch_content: false` and disable the verify pass with `verify_alignment_range: [0, 0]`. Accuracy drops but cost halves.

## What this is not

- Not a content scraper. Bring your own crawl data (Screaming Frog, Sitebulb, custom). The script will fetch URLs as a fallback, but for thousands of pages you want a proper crawler.
- Not a full audit pipeline. This is the relevance-scoring layer. Pair it with GSC + Ahrefs data for the full prune-or-keep decision.
- Not for affiliate sites or pure publishers — yet. The taxonomy assumes a service or product business. PRs welcome.

## License

MIT. Built by [Lum Kamishi](https://lumkamishi.com).
