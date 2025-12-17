import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import csv
import base64
import requests
import json
import re
from datetime import datetime
import ctypes

# Workaround for WinError 1114: Load libiomp5md.dll before importing torch
try:
    ctypes.CDLL(r"C:\Users\Aqib Rehman\AppData\Local\Programs\Python\Python314\Lib\site-packages\torch\lib\libiomp5md.dll")
except Exception:
    pass

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Diabetes Concept Matcher", page_icon="🩸", layout="wide")

BASE_DIR = "./diabetes_subset_rf2"
NEW_TERMS_LOG = "new_terms_log.csv"
UMLS_FILE = f"{BASE_DIR}/UMLS_DIABETES_CLEAN.tsv"

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.70

DESC_PATH = f"{BASE_DIR}/descriptions_diabetes.tsv"
CON_PATH = f"{BASE_DIR}/concepts_diabetes.tsv"
REL_PATH = f"{BASE_DIR}/relationships_diabetes.tsv"


# ============================================================
# SESSION STATE
# ============================================================
if "queued_phrases" not in st.session_state:
    st.session_state["queued_phrases"] = set()


# ============================================================
# LOAD MODELS (CACHED)
# ============================================================
@st.cache_resource
def load_models():
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    sap_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sap_tokenizer = AutoTokenizer.from_pretrained(sap_name)
    sap_model = AutoModel.from_pretrained(sap_name)
    sap_model.eval()

    return zero_shot, sap_tokenizer, sap_model


zero_shot, sap_tokenizer, sap_model = load_models()


# ============================================================
# EMBEDDING
# ============================================================
def embed_sapbert(texts):
    inputs = sap_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,  # stable + removes HF warning
        return_tensors="pt",
    )
    with torch.no_grad():
        out = sap_model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_rf2(base_dir):
    desc = pd.read_csv(DESC_PATH, sep="\t", dtype=str)
    con = pd.read_csv(CON_PATH, sep="\t", dtype=str)
    rel = pd.read_csv(REL_PATH, sep="\t", dtype=str)

    desc = desc[desc["active"] == "1"]

    terms = desc["term"].dropna().astype(str).unique().tolist()
    ids = desc["conceptId"].astype(str).tolist()

    return desc, con, rel, terms, ids


descriptions, concepts, relationships, all_terms, concept_ids = load_rf2(BASE_DIR)


@st.cache_resource
def get_term_embeddings(terms):
    if not terms:
        return np.zeros((0, 768), dtype="float32")
    return embed_sapbert(terms)


term_embeddings = get_term_embeddings(all_terms)


@st.cache_data
def load_umls_subset(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["CUI", "STR", "SAB", "TTY"]), [], []

    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.dropna(subset=["CUI", "STR"])
    df["CUI"] = df["CUI"].astype(str)
    df["STR"] = df["STR"].astype(str)

    return df, df["STR"].tolist(), df["CUI"].tolist()


umls_df, umls_terms, umls_cuis = load_umls_subset(UMLS_FILE)


@st.cache_resource
def get_umls_embeddings(terms):
    if not terms:
        return np.zeros((0, 768), dtype="float32")
    return embed_sapbert(terms)


umls_embeddings = get_umls_embeddings(umls_terms)


# ============================================================
# TEXT NORMALIZATION (OPTION C)
# ============================================================
MEDICAL_ABBREVS = {
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "htn": "hypertension",
    "bg": "blood glucose",
    "bs": "blood sugar",
    "hba1c": "hemoglobin a1c",
}

STOPWORDS = {
    "i", "im", "me", "my", "mine", "you", "u", "your", "yours",
    "he", "she", "they", "them", "we", "us", "our", "ours",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    "feel", "feeling", "felt", "get", "got", "getting",
    "a", "an", "the", "this", "that", "these", "those",
    "very", "so", "too", "much", "lot", "little", "bit",
    "of", "in", "on", "at", "for", "from", "to", "with",
    "without", "after", "before", "during", "since", "because",
    "but", "if", "when", "while", "and", "or", "as", "like",
    "just", "today", "yesterday", "tomorrow", "now", "right",
    "really", "also", "again", "please", "plz", "kindly",
}


@st.cache_resource
def build_medical_vocab(snomed_terms, umls_terms, abbrev_map):
    vocab = set()

    def add_terms(term_list):
        for t in term_list:
            for w in re.findall(r"[a-z0-9]+", str(t).lower()):
                vocab.add(w)

    add_terms(snomed_terms)
    add_terms(umls_terms)

    for v in abbrev_map.values():
        for w in re.findall(r"[a-z0-9]+", v.lower()):
            vocab.add(w)

    return vocab


medical_vocab = build_medical_vocab(all_terms, umls_terms, MEDICAL_ABBREVS)


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()

    for abbr, full in MEDICAL_ABBREVS.items():
        t = re.sub(rf"\b{re.escape(abbr)}\b", full, t)

    t = re.sub(r"[^a-z0-9\s]", " ", t)

    tokens = t.split()
    cleaned = []

    for tok in tokens:
        if tok in STOPWORDS:
            continue

        if re.match(r"^\d+(\.\d+)?(mg|mmol|ml|mgdl|mmoll)?$", tok):
            cleaned.append(tok)
            continue

        if tok in medical_vocab:
            cleaned.append(tok)
            continue

        if re.match(r"^[a-z]{6,}$", tok):
            cleaned.append(tok)

    if not cleaned:
        return text.lower().strip()

    return " ".join(cleaned)


# ============================================================
# DIABETES RELEVANCE (ZERO-SHOT + CENTROID)
# ============================================================
diabetes_anchors = [
    "diabetes",
    "diabetes mellitus",
    "type 2 diabetes mellitus",
    "type 1 diabetes mellitus",
    "hyperglycemia",
    "hypoglycemia",
    "blood glucose",
    "insulin resistance",
]
diabetes_centroid = embed_sapbert(diabetes_anchors).mean(axis=0, keepdims=True)


def is_diabetes_related(text_clean: str):
    t = text_clean.lower().strip()

    zs = zero_shot(t, ["diabetes", "not_related"])
    zs_label = zs["labels"][0]
    zs_score = zs["scores"][0]

    if zs_label == "diabetes" and zs_score >= REVIEW_THRESHOLD:
        return True, "zero-shot", float(zs_score)

    sim = float(cosine_similarity(embed_sapbert([t]), diabetes_centroid)[0][0])
    if sim >= REVIEW_THRESHOLD:
        return True, "centroid", sim

    return False, "none", float(max(zs_score, sim))


# ============================================================
# UMLS MATCH HELPERS
# ============================================================
def umls_exact_match(text_clean: str):
    if umls_df.empty:
        return None

    mask = umls_df["STR"].str.strip().str.lower() == text_clean
    if not mask.any():
        return None

    row = umls_df[mask].iloc[0]
    return {"term": row["STR"], "cui": row["CUI"], "similarity": 1.0, "mode": "exact"}


def umls_embedding_match(text_clean: str):
    if umls_embeddings.shape[0] == 0:
        return None

    emb = embed_sapbert([text_clean])
    sims = cosine_similarity(emb, umls_embeddings)[0]
    idx = int(np.argmax(sims))

    return {
        "term": umls_terms[idx],
        "cui": umls_cuis[idx],
        "similarity": float(sims[idx]),
        "mode": "embedding",
    }


# ============================================================
# LOGGING
# ============================================================
def init_new_terms_log():
    if not os.path.exists(NEW_TERMS_LOG):
        with open(NEW_TERMS_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "raw_phrase",
                "clean_phrase",
                "matched_term",
                "concept_id",
                "source",
                "similarity",
                "action",
                "review_time",
            ])


def log_new_term(raw, clean, term, cid, source, sim, action):
    init_new_terms_log()

    if os.path.exists(NEW_TERMS_LOG):
        df = pd.read_csv(NEW_TERMS_LOG)
        if raw.lower().strip() in df["raw_phrase"].astype(str).str.lower().values and action == "queued":
            return

    with open(NEW_TERMS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            raw,
            clean,
            term,
            cid,
            source,
            f"{float(sim):.3f}",
            action,
            "",
        ])


# ============================================================
# ADD SYNONYM TO SNOMED SUBSET
# ============================================================
def append_new_phrase_to_subset(raw_phrase, parent_concept_id):
    global descriptions, all_terms, concept_ids, term_embeddings

    df = pd.read_csv(DESC_PATH, sep="\t", dtype=str)

    clean = preprocess_text(raw_phrase)
    df_terms = df["term"].astype(str).str.strip().str.lower()

    if clean in df_terms.values:
        return False

    new_row = {
        "id": f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "effectiveTime": datetime.now().strftime("%Y%m%d"),
        "active": "1",
        "moduleId": "999999999999999999",
        "conceptId": parent_concept_id,
        "languageCode": "en",
        "typeId": "900000000000003001",
        "term": raw_phrase,
        "caseSignificanceId": "900000000000020002",
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DESC_PATH, sep="\t", index=False)

    descriptions = pd.concat([descriptions, pd.DataFrame([new_row])], ignore_index=True)

    all_terms.append(raw_phrase)
    concept_ids.append(parent_concept_id)

    new_emb = embed_sapbert([raw_phrase])
    if term_embeddings.shape[0] == 0:
        term_embeddings = new_emb
    else:
        term_embeddings = np.vstack([term_embeddings, new_emb])

    return True


# ============================================================
# MATCHING PIPELINE
# ============================================================
def analyze_user_phrase(raw_text: str, desc_df: pd.DataFrame):
    clean = preprocess_text(raw_text)

    # 1) exact SNOMED synonym -> 1.0
    df_terms = desc_df["term"].astype(str).str.strip().str.lower()
    if clean in df_terms.values:
        row = desc_df[df_terms == clean].iloc[0]
        cid = row["conceptId"]
        return {
            "diabetes_related": True,
            "clean": clean,
            "rel_method": "exact-snomed-synonym",
            "relevance_score": 1.0,
            "decision_class": "auto",
            "final": {"source": "SNOMED", "term": row["term"], "id": cid, "sim": 1.0},
            "snomed": {"term": row["term"], "id": cid, "sim": 1.0, "decision": "Exact synonym"},
            "umls": None,
        }

    # 2) diabetes relevance
    related, rel_method, rel_score = is_diabetes_related(clean)
    if not related:
        return {
            "diabetes_related": False,
            "reason": f"Not diabetes-related (score={rel_score:.3f}, method={rel_method})",
        }

    # 3) SNOMED embedding
    user_emb = embed_sapbert([clean])

    sn_term, sn_id, sn_sim = None, None, 0.0
    if term_embeddings.shape[0] > 0:
        sn_sims = cosine_similarity(user_emb, term_embeddings)[0]
        sn_i = int(np.argmax(sn_sims))
        sn_term = all_terms[sn_i]
        sn_id = concept_ids[sn_i]
        sn_sim = float(sn_sims[sn_i])

    if sn_sim >= AUTO_THRESHOLD:
        sn_dec = "High SNOMED match (≥ 0.80)"
    elif sn_sim >= REVIEW_THRESHOLD:
        sn_dec = "Borderline SNOMED match (0.70–0.79)"
    else:
        sn_dec = "Low SNOMED similarity (< 0.70)"

    # 4) UMLS fallback
    um = None
    um_sim = -1.0

    um_exact = umls_exact_match(clean)
    if um_exact is not None:
        um = um_exact
        um_sim = 1.0
    else:
        um_emb = umls_embedding_match(clean)
        if um_emb is not None:
            um = um_emb
            um_sim = float(um_emb["similarity"])

    # 5) final choice
    if sn_sim >= um_sim:
        final_source = "SNOMED"
        final_term = sn_term
        final_id = sn_id
        final_sim = sn_sim
    else:
        final_source = "UMLS"
        final_term = um["term"] if um else None
        final_id = um["cui"] if um else None
        final_sim = um_sim

    # 6) decision class
    if final_term is None:
        decision_class = "discard"
    elif final_sim >= AUTO_THRESHOLD:
        decision_class = "auto"
    elif final_sim >= REVIEW_THRESHOLD:
        decision_class = "review"
    else:
        decision_class = "discard"

    return {
        "diabetes_related": True,
        "clean": clean,
        "rel_method": rel_method,
        "relevance_score": rel_score,
        "decision_class": decision_class,
        "final": {"source": final_source, "term": final_term, "id": final_id, "sim": float(final_sim)},
        "snomed": {"term": sn_term, "id": sn_id, "sim": float(sn_sim), "decision": sn_dec},
        "umls": um,
    }


# ============================================================
# UI
# ============================================================
st.title("🩸 Diabetes Concept Matcher (SNOMED CT + UMLS)")
st.caption("Run locally in Antigravity IDE:  streamlit run app.py")

colA, colB = st.columns([2, 1])

with colA:
    user_input = st.text_input("Enter a phrase:", placeholder="e.g., my sugar is high today, shaky after meals")
    run_btn = st.button("🔎 Analyze Phrase")

with colB:
    st.markdown("### Thresholds")
    st.write(f"✅ Auto-map: **≥ {AUTO_THRESHOLD:.2f}**")
    st.write(f"🔎 Review: **{REVIEW_THRESHOLD:.2f}–{AUTO_THRESHOLD-0.01:.2f}**")
    st.write(f"❌ Discard: **< {REVIEW_THRESHOLD:.2f}**")


if run_btn and user_input.strip():
    res = analyze_user_phrase(user_input, descriptions)

    if not res["diabetes_related"]:
        st.error(res["reason"])
    else:
        st.success(f"Diabetes-related ({res['rel_method']}, score={res['relevance_score']:.3f})")
        st.write(f"**Cleaned input:** `{res['clean']}`")

        left, right = st.columns(2)

        with left:
            st.subheader("SNOMED CT")
            sn = res["snomed"]
            st.write(f"Decision: **{sn['decision']}**")
            st.write(f"Concept: **{sn['term']}**")
            st.write(f"Concept ID: `{sn['id']}`")
            st.write(f"Similarity: `{sn['sim']:.3f}`")

        with right:
            st.subheader("UMLS (Fallback)")
            if res["umls"] is None:
                st.info("No UMLS candidate (or file missing).")
            else:
                st.write(f"Term: **{res['umls']['term']}**")
                st.write(f"CUI: `{res['umls']['cui']}`")
                st.write(f"Similarity: `{float(res['umls']['similarity']):.3f}`")
                st.write(f"Mode: `{res['umls']['mode']}`")

        st.subheader("Final Suggestion")
        f = res["final"]
        st.success(f"Source: **{f['source']}** | Concept: **{f['term']}** (`{f['id']}`) | Similarity: `{f['sim']:.3f}`")

        if res["decision_class"] == "auto":
            st.info("Auto-map path (≥ 0.80). You may add it as synonym if SNOMED source.")
            if f["source"] == "SNOMED":
                if st.checkbox("➕ Add as new SNOMED synonym"):
                    ok = append_new_phrase_to_subset(user_input, f["id"])
                    if ok:
                        st.success("Added to SNOMED subset.")
                        log_new_term(user_input, res["clean"], f["term"], f["id"], f["source"], f["sim"], "added")
                    else:
                        st.warning("Already exists in subset.")
        elif res["decision_class"] == "review":
            st.warning("Review path (0.70–0.79). Queue for admin review.")
            if st.checkbox("Queue for review"):
                key = user_input.lower().strip()
                if key not in st.session_state["queued_phrases"]:
                    log_new_term(user_input, res["clean"], f["term"], f["id"], f["source"], f["sim"], "queued")
                    st.session_state["queued_phrases"].add(key)
                    st.success("Queued.")
                else:
                    st.info("Already queued.")
        else:
            st.error("Discarded (<0.70).")
            log_new_term(user_input, res["clean"], f.get("term", "-"), f.get("id", "-"), f.get("source", "-"), f.get("sim", 0.0), "discarded")


# ============================================================
# ADMIN REVIEW PANEL
# ============================================================
st.divider()
st.header("🔐 Admin Review")

with st.expander("Open Admin Panel (Approve / Reject)"):
    if not os.path.exists(NEW_TERMS_LOG):
        st.info("No log file yet.")
    else:
        df = pd.read_csv(NEW_TERMS_LOG)

        # keep latest per raw_phrase
        df["lower"] = df["raw_phrase"].astype(str).str.lower()
        df = df.sort_values("timestamp")
        df = df[~df["lower"].duplicated(keep="last")].drop(columns=["lower"])
        df.to_csv(NEW_TERMS_LOG, index=False)

        pending = df[df["action"] == "queued"].reset_index(drop=True)

        if pending.empty:
            st.info("No queued items.")
        else:
            for i, row in pending.iterrows():
                phrase = row["raw_phrase"]
                clean = row["clean_phrase"]
                term = row["matched_term"]
                cid = row["concept_id"]
                source = row["source"]
                score = row["similarity"]

                st.markdown(f"### Phrase: {phrase}")
                st.write(f"Clean: `{clean}`")
                st.write(f"Suggested: **{term}** | ID: `{cid}` | Source: **{source}** | Similarity: `{score}`")

                c1, c2 = st.columns(2)
                if c1.button("Approve", key=f"ap_{i}"):
                    # Only add to SNOMED subset if concept_id looks like a SNOMED conceptId
                    # If source==UMLS, we still approve but we do NOT create a new SNOMED concept here
                    if source == "SNOMED" and isinstance(cid, str) and cid.strip() != "":
                        ok = append_new_phrase_to_subset(phrase, cid)
                        if ok:
                            st.success("Approved and added as SNOMED synonym.")
                        else:
                            st.warning("Already exists.")
                    else:
                        st.info("Approved. (Source is UMLS; no SNOMED synonym added.)")

                    df.loc[df["raw_phrase"] == phrase, "action"] = "approved"
                    df.loc[df["raw_phrase"] == phrase, "review_time"] = datetime.now().isoformat(timespec="seconds")
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.rerun()

                if c2.button("Reject", key=f"rej_{i}"):
                    df.loc[df["raw_phrase"] == phrase, "action"] = "rejected"
                    df.loc[df["raw_phrase"] == phrase, "review_time"] = datetime.now().isoformat(timespec="seconds")
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.warning("Rejected.")
                    st.rerun()


# ============================================================
# KNOWLEDGE GRAPH + ANALYTICS
# ============================================================
st.divider()
st.header("📡 Knowledge Graph & Analytics")

desc_df = pd.read_csv(DESC_PATH, sep="\t", dtype=str)
rel_df = pd.read_csv(REL_PATH, sep="\t", dtype=str)
desc_df = desc_df[desc_df["active"] == "1"]
rel_df = rel_df[rel_df["active"] == "1"]

G = nx.DiGraph()
for cid in desc_df["conceptId"].unique():
    G.add_node(cid)

isa_edges = rel_df[rel_df["typeId"] == "116680003"]  # IS-A
for _, r in isa_edges.iterrows():
    parent = r["destinationId"]
    child = r["sourceId"]
    if parent in G.nodes and child in G.nodes:
        G.add_edge(parent, child)

new_synonyms = desc_df[desc_df["id"].str.contains("user_", na=False)]

with st.expander("🧠 Show Knowledge Graph"):
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightblue", ax=ax)

    # Highlight concepts that received new synonyms
    new_parents = new_synonyms["conceptId"].unique()
    nx.draw_networkx_nodes(G, pos, nodelist=new_parents, node_color="orange", node_size=200, ax=ax)

    nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3, ax=ax)
    ax.set_title("SNOMED Diabetes Subset Graph\nOrange = Concepts with New Synonyms")
    ax.axis("off")
    st.pyplot(fig)

    st.info(f"New synonyms added: **{len(new_synonyms)}**")
    st.info(f"Orange parent concepts count: **{len(new_parents)}**")

with st.expander("📈 Analytics / Stats"):
    if os.path.exists(NEW_TERMS_LOG):
        log_df = pd.read_csv(NEW_TERMS_LOG)

        st.write("### Log Summary")
        st.metric("Added", int((log_df["action"] == "added").sum()))
        st.metric("Queued", int((log_df["action"] == "queued").sum()))
        st.metric("Approved", int((log_df["action"] == "approved").sum()))
        st.metric("Rejected", int((log_df["action"] == "rejected").sum()))
        st.metric("Discarded", int((log_df["action"] == "discarded").sum()))

        st.write("### Activity Trend")
        if not log_df.empty:
            log_df["day"] = pd.to_datetime(log_df["timestamp"]).dt.date
            trend = log_df.groupby("day").size()

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(trend.index, trend.values)
            ax2.set_title("Daily Activity Trend")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Actions")
            st.pyplot(fig2)

        st.write("### Full Log Preview")
        st.dataframe(log_df.tail(50), use_container_width=True)
    else:
        st.info("No analytics yet.")


# ============================================================
# DOWNLOAD LOG FILE
# ============================================================
st.divider()
st.header("📥 Downloads")

init_new_terms_log()
if os.path.exists(NEW_TERMS_LOG):
    with open(NEW_TERMS_LOG, "rb") as f:
        st.download_button("Download Log CSV", f, file_name="new_terms_log.csv")


# ============================================================
# ZIP DOWNLOAD OF SUBSET
# ============================================================
def make_zip():
    zip_path = f"{BASE_DIR}/subset_updated.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ["descriptions_diabetes.tsv", "concepts_diabetes.tsv", "relationships_diabetes.tsv"]:
            p = f"{BASE_DIR}/{fn}"
            if os.path.exists(p):
                z.write(p, arcname=fn)
    return zip_path


if st.button("Create ZIP of Updated Subset"):
    zp = make_zip()
    with open(zp, "rb") as f:
        st.download_button("Download Updated Subset ZIP", f, file_name="subset_updated.zip")


# ============================================================
# PUSH TO GITHUB
# ============================================================
st.divider()
st.header("🚀 Push Updated TSV to GitHub")

with st.expander("GitHub Push Settings"):
    repo = st.text_input("Repository (e.g., username/diabetes-matcher-ui)")
    token = st.text_input("GitHub Token", type="password")

    if st.button("Push descriptions_diabetes.tsv to GitHub"):
        if not repo or not token:
            st.error("Missing repo or token.")
        else:
            api_url = f"https://api.github.com/repos/{repo}/contents/diabetes_subset_rf2/descriptions_diabetes.tsv"

            with open(DESC_PATH, "rb") as f:
                content = base64.b64encode(f.read()).decode()

            headers = {"Authorization": f"token {token}"}
            resp = requests.get(api_url, headers=headers)
            sha = resp.json().get("sha") if resp.status_code == 200 else None

            payload = {"message": "Auto-update from Streamlit app", "content": content, "branch": "main"}
            if sha:
                payload["sha"] = sha

            r = requests.put(api_url, headers=headers, data=json.dumps(payload))

            if r.status_code in [200, 201]:
                st.success("Pushed to GitHub successfully!")
            else:
                st.error(f"GitHub Error: {r.text}")
