"""
Topical proximity scoring for content audits.

Generic LLM-based classifier that scores blog posts against any site's
service offerings. Replaces embedding similarity with multi-dimensional
classification: post_type, commercial_intent, service_alignment, tier.

Supports OpenAI and Anthropic (Claude) as providers. Switch via config.

Why LLM > embeddings here:
  Embeddings collapse "solar installation cost" and "Powerwall review"
  to the same neighborhood -- both are "about solar." Classification
  separates what the post IS from what it's about.

Inputs:
  config.yaml      business context + provider/model choice
  services.csv     cols: Service_Name (req), Descriptor (req)
  audit.csv        cols: URL (req), Topic/Title/H1/Meta Description (any)
                         Content (opt -- fetched if missing)

Output:
  topical_proximity.csv -- one row per input, with classification cols.

Setup:
  pip install openai anthropic pandas pyyaml httpx trafilatura tenacity
  export OPENAI_API_KEY=sk-...           # if using openai
  export ANTHROPIC_API_KEY=sk-ant-...    # if using anthropic
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
import trafilatura
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential


# ----------------------------- Config -----------------------------

DEFAULT_CONFIG = {
    "business": {
        "name": "Example Co",
        "type": "service business",
        "location": "",
        "description": "Edit config.yaml to describe what this business does.",
    },
    "classification": {
        "provider": "anthropic",          # "openai" or "anthropic"
        "model": "claude-sonnet-4-6",            # provider-appropriate model id
        "workers": 2,
        "fetch_content": True,
        "content_max_chars": 4000,
        "verify_alignment_range": [0.35, 0.65],
        "checkpoint_every": 25,
    },
}


def load_config(path: str) -> dict:
    if not Path(path).exists():
        return DEFAULT_CONFIG
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    return {
        "business": {**DEFAULT_CONFIG["business"], **user_cfg.get("business", {})},
        "classification": {**DEFAULT_CONFIG["classification"], **user_cfg.get("classification", {})},
    }


# ----------------------------- Provider clients -----------------------------

_openai_client = None
_anthropic_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")
        _openai_client = OpenAI()
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running.")
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


# ----------------------------- Content fetching -----------------------------

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def fetch_content(url: str, max_chars: int) -> str:
    """Fetch a URL and extract main content. Returns empty string on failure."""
    try:
        with httpx.Client(timeout=15, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (compatible; ContentAudit/1.0)"}) as c:
            r = c.get(url)
            r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        return text[:max_chars].strip()
    except Exception:
        return ""


# ----------------------------- Tier rules -----------------------------

def compute_tier(c: dict) -> str:
    """Deterministic tier mapping applied AFTER classification."""
    if c.get("post_type") == "OFF_TOPIC" or (c.get("service") or "").upper() == "NONE":
        return "OFF_TOPIC"

    sa = c.get("service_alignment") or 0
    pt = c.get("post_type", "")
    ci = c.get("commercial_intent", "LOW")

    if pt == "SERVICE_INTENT" and ci in ("HIGH", "MEDIUM"):
        return "CORE_SERVICE"
    if pt == "PRODUCT_REVIEW" and sa >= 0.5:
        return "CORE_PRODUCT"
    if pt == "COMPARISON" and sa >= 0.5:
        return "ADJACENT_COMPARISON"
    if pt in ("EDUCATIONAL", "INSTALLATION_GUIDE") and sa >= 0.4:
        return "ADJACENT_INFORMATIONAL"
    if pt == "DESIGN_INSPIRATION":
        return "ADJACENT_INFORMATIONAL" if sa >= 0.4 else "PERIPHERAL"
    if pt == "DRIFT" or sa < 0.3:
        return "PERIPHERAL"
    return "ADJACENT_INFORMATIONAL"


# ----------------------------- Few-shot examples -----------------------------
# Mixed-domain examples so the model learns the PATTERN, not the industry.

FEW_SHOT_EXAMPLES = [
    {
        "input": "Title: How Much Does Solar Installation Cost in 2025?\nH1: Solar Installation Cost Guide",
        "output": {
            "reasoning": "Cost query for a service the site offers -- classic hire-me intent.",
            "service": "Residential Solar Installation",
            "service_alignment": 0.95,
            "post_type": "SERVICE_INTENT",
            "commercial_intent": "HIGH",
        },
    },
    {
        "input": "Title: Tesla Powerwall 3 Review: Is It Worth It?\nH1: Powerwall 3 In-Depth Review",
        "output": {
            "reasoning": "Product-specific review of a brand within an offered service category. Not hire-me intent but supports the service decision.",
            "service": "Battery Backup Systems",
            "service_alignment": 0.7,
            "post_type": "PRODUCT_REVIEW",
            "commercial_intent": "MEDIUM",
        },
    },
    {
        "input": "Title: Trex vs Fiberon: Which Composite Decking Wins?\nH1: Trex vs Fiberon",
        "output": {
            "reasoning": "Brand-vs-brand comparison within an offered service area. Mid-funnel decision content.",
            "service": "Deck Building",
            "service_alignment": 0.65,
            "post_type": "COMPARISON",
            "commercial_intent": "MEDIUM",
        },
    },
    {
        "input": "Title: 10 Modern Kitchen Color Trends for 2025\nH1: Kitchen Color Trends",
        "output": {
            "reasoning": "Aesthetic inspiration content. Loosely supports kitchen remodel but no decision or hire intent.",
            "service": "Kitchen Remodeling",
            "service_alignment": 0.35,
            "post_type": "DESIGN_INSPIRATION",
            "commercial_intent": "LOW",
        },
    },
    {
        "input": "Title: Best LED Bulbs for Your Kitchen\nH1: Kitchen Lighting Guide",
        "output": {
            "reasoning": "Adjacent to kitchen remodel but about a product the business does not sell or install. Drift.",
            "service": "NONE",
            "service_alignment": 0.15,
            "post_type": "DRIFT",
            "commercial_intent": "LOW",
        },
    },
    {
        "input": "Title: How to Install Hardie Siding (Step-by-Step)\nH1: DIY Hardie Installation",
        "output": {
            "reasoning": "Process content within the service area. Educational, supports the service indirectly.",
            "service": "Siding Replacement",
            "service_alignment": 0.5,
            "post_type": "INSTALLATION_GUIDE",
            "commercial_intent": "LOW",
        },
    },
]


# ----------------------------- Schemas -----------------------------
# Same schema for both providers. OpenAI uses it as response_format json_schema.
# Anthropic uses it as a tool input_schema.

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Reason through what this post IS before classifying. 2-3 sentences.",
        },
        "service": {
            "type": "string",
            "description": "The service this post supports, or 'NONE'.",
        },
        "service_alignment": {
            "type": "number",
            "description": "0.0-1.0 score for how directly this supports the service.",
        },
        "post_type": {
            "type": "string",
            "enum": ["SERVICE_INTENT", "PRODUCT_REVIEW", "COMPARISON",
                     "INSTALLATION_GUIDE", "EDUCATIONAL", "DESIGN_INSPIRATION",
                     "DRIFT", "OFF_TOPIC"],
        },
        "commercial_intent": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
    },
    "required": ["reasoning", "service", "service_alignment", "post_type", "commercial_intent"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["CONFIRM", "OVERRIDE"]},
        "reasoning": {"type": "string"},
        "service": {"type": "string"},
        "service_alignment": {"type": "number"},
        "post_type": {"type": "string", "enum": [
            "SERVICE_INTENT", "PRODUCT_REVIEW", "COMPARISON",
            "INSTALLATION_GUIDE", "EDUCATIONAL", "DESIGN_INSPIRATION",
            "DRIFT", "OFF_TOPIC"]},
        "commercial_intent": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    },
    "required": ["verdict", "reasoning", "service", "service_alignment", "post_type", "commercial_intent"],
    "additionalProperties": False,
}


# ----------------------------- Prompts -----------------------------

def build_system_prompt(business: dict, services_df: pd.DataFrame) -> str:
    services_block = "\n\n".join(
        f"**{r.Service_Name}**: {r.Descriptor}" for r in services_df.itertuples()
    )

    location_str = f" in {business['location']}" if business.get("location") else ""

    examples_block = "\n\n".join(
        f"--- Example ---\n{ex['input']}\nOutput: {json.dumps(ex['output'])}"
        for ex in FEW_SHOT_EXAMPLES
    )

    return f"""You classify blog posts on a content site to determine how each one supports the business's actual service offerings. The goal is to power a content audit / pruning decision.

Business: {business['name']}, a {business['type']}{location_str}.
{business['description']}

Services offered:

{services_block}

Classify each post on five dimensions:

1. **reasoning** -- 2-3 sentences. What is this post? Who is the reader? What action would they take next? Reason BEFORE deciding. Do not skip this.

2. **service** -- exact service name from above, or "NONE" if the post doesn't relate to any offered service.

3. **service_alignment** (0.0-1.0):
   - 0.9-1.0: hire/cost/process for the service itself (e.g., "installation cost", "find a contractor")
   - 0.6-0.8: product/brand content within the service category
   - 0.4-0.6: educational about service-relevant topics
   - 0.2-0.4: tangentially related
   - 0.0-0.2: off-strategy

4. **post_type** -- pick exactly one, in this priority order:
   - SERVICE_INTENT: about hiring, cost, timeline, or finding a contractor for the service
   - PRODUCT_REVIEW: brand or product-specific review or overview. USE THIS even when the brand relates to an offered service. SERVICE_INTENT is for hire-me content only.
   - COMPARISON: comparing brands, products, or options
   - INSTALLATION_GUIDE: step-by-step process content
   - EDUCATIONAL: top-of-funnel knowledge with no process steps
   - DESIGN_INSPIRATION: aesthetic or visual content
   - DRIFT: adjacent to the industry but not service-related
   - OFF_TOPIC: doesn't belong on this site

5. **commercial_intent**:
   - HIGH: close to hiring (service queries, cost, "near me", contractor)
   - MEDIUM: actively researching (brand comparisons, decision content)
   - LOW: top-of-funnel browsing (inspiration, general education)

Critical rule: a brand or product post is PRODUCT_REVIEW even if the brand relates to a service the business offers. Reserve SERVICE_INTENT for hire-me content (cost, timeline, finding a contractor, process of getting the service done).

Examples:

{examples_block}
"""


def build_verification_prompt() -> str:
    return """You verify a prior classification of a blog post. The prior model already classified it. Your job: independently re-examine the post against the same criteria, then either CONFIRM the prior verdict or OVERRIDE it with a corrected one.

Rules:
- Only override if you have a clear reason rooted in the post content
- Pay special attention to PRODUCT_REVIEW vs SERVICE_INTENT distinction
- Pay special attention to DRIFT vs OFF_TOPIC distinction
- If overriding, supply the corrected fields"""


# ----------------------------- Unified LLM call -----------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_llm_structured(provider: str, model: str, system: str, user_msg: str,
                          schema: dict, schema_name: str) -> dict:
    """Single entry point. Dispatches to OpenAI or Anthropic and returns
    a parsed dict matching the schema."""

    if provider == "openai":
        client = _get_openai()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            max_completion_tokens=1500,
        )
        return json.loads(resp.choices[0].message.content)

    elif provider == "anthropic":
        client = _get_anthropic()
        tool_def = {
            "name": schema_name,
            "description": f"Return a structured {schema_name} for the given input.",
            "input_schema": schema,
        }
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": schema_name},
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == schema_name:
                return dict(block.input)
        raise RuntimeError(f"Anthropic response did not contain a {schema_name} tool_use block.")

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'anthropic'.")


# ----------------------------- Per-row classification -----------------------------

def _build_user_msg(post: dict) -> str:
    parts = []
    for col in ["Topic", "Title", "H1", "Meta Description"]:
        v = post.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
            parts.append(f"{col}: {str(v).strip()}")
    content = post.get("Content") or ""
    if content:
        parts.append(f"\nPage content (truncated):\n{content}")
    return "\n".join(parts)


def classify_post(provider: str, model: str, system_prompt: str, post: dict) -> dict:
    user_msg = _build_user_msg(post)
    return _call_llm_structured(provider, model, system_prompt, user_msg,
                                  CLASSIFICATION_SCHEMA, "classification")


def verify_classification(provider: str, model: str, system_prompt: str,
                           post: dict, prior: dict) -> dict:
    user_msg = _build_user_msg(post)
    user_msg += f"\n\nPrior classification:\n{json.dumps(prior, indent=2)}"
    return _call_llm_structured(provider, model, system_prompt, user_msg,
                                  VERIFICATION_SCHEMA, "verification")


# ----------------------------- Pipeline -----------------------------

def run(config_path: str = "config.yaml",
        services_path: str = "services.csv",
        audit_path: str = "audit.csv",
        output_path: str = "topical_proximity.csv") -> pd.DataFrame:

    cfg = load_config(config_path)
    cls_cfg = cfg["classification"]
    provider = cls_cfg["provider"]
    model = cls_cfg["model"]

    print(f"Provider: {provider} | Model: {model}")

    services = pd.read_csv(services_path).dropna(subset=["Service_Name", "Descriptor"])
    audit = pd.read_csv(audit_path).reset_index(drop=True).copy()

    # ---- Stage 1: fetch content ----
    if cls_cfg["fetch_content"] and "URL" in audit.columns:
        if "Content" not in audit.columns:
            audit["Content"] = ""
        needs_fetch = audit[audit["Content"].fillna("").str.len() < 100]
        print(f"\nFetching content for {len(needs_fetch)} URLs...")
        with ThreadPoolExecutor(max_workers=cls_cfg["workers"]) as pool:
            futs = {pool.submit(fetch_content, row.URL, cls_cfg["content_max_chars"]): i
                    for i, row in needs_fetch.iterrows() if pd.notna(row.URL)}
            for fut in as_completed(futs):
                audit.at[futs[fut], "Content"] = fut.result()
        n_fetched = (audit["Content"].fillna("").str.len() >= 100).sum()
        print(f"  Got content for {n_fetched}/{len(audit)} posts.")

    system_prompt = build_system_prompt(cfg["business"], services)

    # ---- Stage 2: classify ----
    print(f"\nClassifying {len(audit)} posts...")
    results = [None] * len(audit)
    with ThreadPoolExecutor(max_workers=cls_cfg["workers"]) as pool:
        futs = {pool.submit(classify_post, provider, model, system_prompt,
                            audit.iloc[i].to_dict()): i for i in range(len(audit))}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"error": str(e)}
            done += 1
            if done % cls_cfg["checkpoint_every"] == 0:
                print(f"  {done}/{len(audit)}")

    # ---- Stage 3: verify ambiguous rows ----
    lo, hi = cls_cfg["verify_alignment_range"]
    ambiguous_idx = [i for i, r in enumerate(results)
                     if r and "error" not in r and lo <= (r.get("service_alignment") or 0) <= hi]
    print(f"\nVerifying {len(ambiguous_idx)} ambiguous rows (alignment {lo}-{hi})...")
    verify_sys = build_verification_prompt()
    with ThreadPoolExecutor(max_workers=cls_cfg["workers"]) as pool:
        futs = {pool.submit(verify_classification, provider, model, verify_sys,
                            audit.iloc[i].to_dict(), results[i]): i for i in ambiguous_idx}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                v = fut.result()
                if v["verdict"] == "OVERRIDE":
                    results[i] = {
                        "reasoning": v["reasoning"],
                        "service": v["service"],
                        "service_alignment": v["service_alignment"],
                        "post_type": v["post_type"],
                        "commercial_intent": v["commercial_intent"],
                        "overridden": True,
                    }
                else:
                    results[i]["overridden"] = False
            except Exception:
                pass

    # ---- Assemble ----
    out = audit.copy()
    for k in ["reasoning", "service", "service_alignment", "post_type",
              "commercial_intent", "overridden", "error"]:
        out[k] = [r.get(k) if r else None for r in results]
    out["tier"] = [compute_tier(r) if r and "error" not in r else "ERROR" for r in results]

    out.to_csv(output_path, index=False)
    print(f"\nSaved {len(out)} rows to {output_path}")
    print("\nTier breakdown:")
    print(out["tier"].value_counts().to_string())
    overrides = out["overridden"].fillna(False).infer_objects(copy=False).sum()
    if overrides:
        print(f"\nOverridden by verification pass: {overrides}")
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(
        prog="topical-proximity",
        description="LLM-based content audit classifier (OpenAI or Anthropic).",
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--services", default="services.csv")
    p.add_argument("--audit", default="audit.csv")
    p.add_argument("--output", default="topical_proximity.csv")
    args = p.parse_args()
    run(config_path=args.config, services_path=args.services,
        audit_path=args.audit, output_path=args.output)


if __name__ == "__main__":
    main()
