# repl.py
"""
Lab04 — CLI REPL for Pandas help (Ollama-only)

Requires in index.py:
  - get_or_create_collection(input_path, model)
  - query_collection(collection, query, n_results=5)

Flow:
  1) init vector collection once
  2) prompt loop
  3) semantic retrieve (top-K)
  4) if weak/empty => print-only "Insufficient context"
  5) else assemble short context and call Ollama LLM
"""

import os, sys
from datetime import datetime
import chromadb

# --- Config (env overrides) ---
INPUT_PATH    = os.environ.get("LAB04_INPUT_PATH", "./data")
EMBED_MODEL   = os.environ.get("LAB04_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_MODEL   = "qwen3-embedding:8b"
# EMBED_MODEL   = "qwen3-embedding:0.6b"
GEN_MODEL     = os.environ.get("LAB04_GEN_MODEL", "gemma3:4b")   # any pulled Ollama model
TOP_K         = int(os.environ.get("LAB04_TOP_K", "5"))
MAX_DISTANCE  = float(os.environ.get("LAB04_MAX_DISTANCE", "0.35"))  # smaller = more similar
# MAX_DISTANCE  = float(os.environ.get("LAB04_MAX_DISTANCE", "0.5"))  # smaller = more similar
MAX_CTX_CHARS = int(os.environ.get("LAB04_MAX_CTX_CHARS", "4500"))

# --- Partner hooks ---
try:
    from index import get_or_create_collection, query_collection
except Exception as e:
    print("ERROR: index.py not importable next to repl.py", file=sys.stderr)
    raise

def _s(x): return "" if x is None else str(x)

def insufficient_context():
    # STRICT print-only per professor
    print("\nInsufficient context.")
    print("Try:")
    print(" • Refine keywords (specific DataFrame/Series ops or column dtypes).")
    print(" • Try related APIs (e.g., Series.value_counts, DataFrame.mode, groupby/agg).")
    print(" • Specify the exact object shape (e.g., 'Series of ints', 'DataFrame with column tip').")

def _has_code(docs):
    needles = ("```", "pd.", "DataFrame(", "Series(", ".loc", ".groupby(", ".merge(", ".value_counts(")
    for d in docs:
        t = _s(d)
        for n in needles:
            if n in t:
                return True
    return False

def _build_context(docs, metas):
    # Compact: [i] symbol\nSignature: ...\nBlurb: ...
    lines, used = [], 0
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        sym  = _s(meta.get("symbol") or meta.get("name") or meta.get("api") or meta.get("source"))
        sig  = _s(meta.get("signature") or "")
        blb  = " ".join(_s(docs[i]).split())
        if len(blb) > 220: blb = blb[:219] + "…"

        block = f"[{i+1}] {sym}\n"
        if sig:
            block += f"Signature: {sig}\n"
        block += f"Blurb: {blb}\n"

        if used + len(block) > MAX_CTX_CHARS: break
        lines.append(block); used += len(block)
    return "\n".join(lines) if lines else "No relevant context retrieved."

def _make_prompt(user_q, ctx, allow_example):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rules = [
        "Use ONLY the context below.",
        "Cite symbols used (e.g., pandas.Series.value_counts).",
        "Keep it concise and clear.",
        "Include exactly ONE runnable Python example." if allow_example else "Do NOT include code; no safe example in context.",
    ]
    return (
        "\n".join(rules)
        + "\n\n=== Context ===\n" + ctx + "\n=== End Context ===\n\n"
        + f"Time: {now}\nUser question: {user_q}\nAnswer:"
    )

def _ollama_reply(prompt_text):
    try:
        import ollama
        r = ollama.chat(
            model=GEN_MODEL,
            messages=[
                    {
                        "role": "system",
                        "content": "You are a precise Pandas TA. Use ONLY the provided context. If context is insufficient, say so briefly and suggest next steps. Cite symbols explicitly. Provide exactly ONE concise, runnable example only if allowed."
                    },
                    {
                        "role":"user",
                        "content":prompt_text
                    }
                ],
            stream=False
        )
        return r["message"]["content"]
    except Exception as e:
        # Fall back without crashing so your partner can see the exact prompt
        return f"(Ollama unavailable: {e})\n\n--- Prompt Sent ---\n{prompt_text}"

def _unpack_results(res):
    """
    Expect Chroma's raw dict result from collection.query(...):
      {
        "ids":        [[...]],
        "documents":  [[str, ...]],
        "metadatas":  [[dict, ...]],
        "distances":  [[float, ...]],
        # (embeddings may also exist)
      }

    Returns:
      docs  : list[str]
      metas : list[dict]
      dists : list[float|None]
    """
    def _to_str(x): return "" if x is None else str(x)

    if not isinstance(res, dict):
        return [], [], []

    docs_ll  = res.get("documents")  or []
    metas_ll = res.get("metadatas")  or []
    dists_ll = res.get("distances")  or []

    # Chroma returns lists-of-lists (one inner list per query)
    docs  = docs_ll[0]  if (isinstance(docs_ll, list)  and docs_ll  and isinstance(docs_ll[0],  list)) else []
    metas = metas_ll[0] if (isinstance(metas_ll, list) and metas_ll and isinstance(metas_ll[0], list)) else []
    dists = dists_ll[0] if (isinstance(dists_ll, list) and dists_ll and isinstance(dists_ll[0], list)) else []

    # print(f"Lengths | docs: {len(docs)}, metas: {len(metas)}, dists: {len(dists)}")

    # If metadata is missing or shorter, pad to match docs
    if not metas or len(metas) != len(docs):
        metas = [{} for _ in range(len(docs))]

    # If distances missing or shorter, pad with None
    if not dists or len(dists) != len(docs):
        dists = [None for _ in range(len(docs))]

    # Normalize types
    docs  = [_to_str(d) for d in docs]
    metas = [m if isinstance(m, dict) else {} for m in metas]
    out_d = []
    for d in dists:
        try:
            out_d.append(float(d) if d is not None else None)
        except Exception:
            out_d.append(None)

    # print(f"Lengths | docs: {len(docs)}, metas: {len(metas)}, dists: {len(dists)}")
    return docs, metas, out_d


def main():
    # 1) init collection
    try:
        col = get_or_create_collection(model=EMBED_MODEL)
        # print(type(col) == chromadb.Collection)
        print(f"Collection ready (model='{EMBED_MODEL}', path='{INPUT_PATH}'). Type 'exit' to quit.")
    except Exception as e:
        print(f"Init error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) REPL
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not q: continue
        if q.lower() in ("exit","quit"):
            print("Bye."); break

        # 3) retrieve
        try:
            try:
                raw = query_collection(col, query=q, n_results=TOP_K)
            except TypeError:
                raw = query_collection(col, query=q)
        except Exception as e:
            print(f"Query error: {e}", file=sys.stderr)
            insufficient_context()
            continue

        docs, metas, dists = _unpack_results(raw)
        if not docs:
            insufficient_context(); continue
        

        # 4) cosine cutoff if distances exist
        try:
            have = isinstance(dists, list) and len(dists)==len(docs)
            # print(f"Have: {have} | len dists: {len(dists)}")
            print(f"Max Dist: {MAX_DISTANCE} | {dists}")
            symbols = [item["symbol"] for item in metas]
            print(f"IDs: {symbols}")
            if have and all((d is None) or (float(d) > MAX_DISTANCE) for d in dists):
                insufficient_context(); continue
        except Exception:
            pass

        # 5) prompt + answer
        ctx = _build_context(docs, metas)
        # print("build context")
        allow_example = _has_code(docs)   # only show code if context contains code-like text
        # print("has code")
        prompt = _make_prompt(q, ctx, allow_example)
        # print("make prompt")
        ans = _ollama_reply(prompt)
        # print("reply")

        print("\n--- Help ---")
        print(ans)
        print("------------")

if __name__ == "__main__":
    main()
