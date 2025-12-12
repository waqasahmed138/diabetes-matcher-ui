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

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Diabetes Concept Matcher", page_icon="🩸")

BASE_DIR = "./diabetes_subset_rf2"
NEW_TERMS_LOG = "new_terms_log.csv"
UMLS_FILE = f"{BASE_DIR}/UMLS_DIABETES_CLEAN.tsv"


# ============================================================
# SESSION STATE
# ============================================================
if "queued_phrases" not in st.session_state:
    st.session_state["queued_phrases"] = set()


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )

    sap_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sap_tokenizer = AutoTokenizer.from_pretrained(sap_name)
    sap_model = AutoModel.from_pretrained(sap_name)
    sap_model.eval()

    return zero_shot, sap_tokenizer, sap_model


zero_shot, sap_tokenizer, sap_model = load_models()


# ============================================================
# SAPBERT EMBEDDING
# ============================================================
def embed_sapbert(texts):
    inputs = sap_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = sap_model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


# ============================================================
# LOAD SNOMED RF2 SUBSET
# ============================================================
@st.cache_data
def load_rf2(base_dir):
    desc = pd.read_csv(
        f"{base_dir}/descriptions_diabetes.tsv",
        sep="\t",
        dtype=str,
    )
    con = pd.read_csv(
        f"{base_dir}/concepts_diabetes.tsv",
        sep="\t",
        dtype=str,
    )
    rel = pd.read_csv(
        f"{base_dir}/relationships_diabetes.tsv",
        sep="\t",
        dtype=str,
    )

    desc = desc[desc["active"] == "1"]

    terms = desc["term"].dropna().astype(str).unique().tolist()
    ids = desc["conceptId"].astype(str).tolist()

    return desc, con, rel, terms, ids


descriptions, concepts, relationships, all_terms, concept_ids = load_rf2(BASE_DIR)


# ============================================================
# PRECOMPUTE SNOMED TERM EMBEDDINGS
# ============================================================
@st.cache_resource
def get_term_embeddings(terms):
    if not terms:
        return np.zeros((0, 768), dtype="float32")
    return embed_sapbert(terms)


term_embeddings = get_term_embeddings(all_terms)


# ============================================================
# LOAD UMLS DIABETES SUBSET
# ============================================================
@st.cache_data
def load_umls_subset(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["CUI", "STR", "SAB", "TTY"]), [], []

    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.dropna(subset=["CUI", "STR"])
    df["STR"] = df["STR"].astype(str)
    df["CUI"] = df["CUI"].astype(str)

    terms = df["STR"].tolist()
    cuis = df["CUI"].tolist()
    return df, terms, cuis


umls_df, umls_terms, umls_cuis = load_umls_subset(UMLS_FILE)


@st.cache_resource
def get_umls_embeddings(terms):
    if not terms:
        return np.zeros((0, 768), dtype="float32")
    return embed_sapbert(terms)


umls_embeddings = get_umls_embeddings(umls_terms)


# ============================================================
# MEDICAL ABBREVIATIONS
# ============================================================
MEDICAL_ABBREVS = {
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "htn": "hypertension",
    "bg": "blood glucose",
    "bs": "blood sugar",
    "hba1c": "hemoglobin a1c",
    "gtt": "glucose tolerance test",
}


# ============================================================
# MEDICAL VOCAB (FROM SNOMED + UMLS + ABBREVS) — OPTION C
# ============================================================
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


# Strong stopword list for aggressive clinical filtering
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
    "bro", "yar", "yaar", "bhai", "g",
    "meri", "mera", "mere", "bohot", "bahut", "zyada", "kam",
}


# ============================================================
# TEXT PREPROCESSING (OPTION C)
# ============================================================
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.lower()

    # expand abbreviations
    for abbr, full in MEDICAL_ABBREVS.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        t = re.sub(pattern, full, t)

    # keep letters, digits and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)

    tokens = t.split()
    cleaned = []

    for tok in tokens:
        if tok in STOPWORDS:
            continue

        # numeric lab values
        if re.match(r"^\d+(\.\d+)?(mg|mmol|ml|mgdl|mmoll)?$", tok):
            cleaned.append(tok)
            continue

        # medical vocab hit
        if tok in medical_vocab:
            cleaned.append(tok)
            continue

        # long token that looks clinical
        if re.match(r"^[a-z]{6,}$", tok):
            cleaned.append(tok)

    if not cleaned:
        return text.lower().strip()

    return " ".join(cleaned)


# ============================================================
# DIABETES CENTROID VECTOR
# ============================================================
diabetes_anchors = [
    "diabetes",
    "diabetes mellitus",
    "type 2 diabetes",
    "type 1 diabetes",
    "hyperglycemia",
    "insulin resistance",
    "glucose intolerance",
    "high blood sugar",
    "low insulin",
    "increased glucose",
]

anchor_embs = embed_sapbert(diabetes_anchors)
diabetes_centroid = anchor_embs.mean(axis=0, keepdims=True)


# ============================================================
# HYBRID DIABETES RELEVANCE CHECK
# ============================================================
def is_diabetes_related(text_clean: str):
    t = text_clean.lower().strip()

    zs = zero_shot(t, ["diabetes", "not_related"])
    zs_label = zs["labels"][0]
    zs_score = zs["scores"][0]

    if zs_label == "diabetes" and zs_score >= 0.70:
        return True, "zero-shot", float(zs_score)

    emb = embed_sapbert([t])
    sim = cosine_similarity(emb, diabetes_centroid)[0][0]

    if sim >= 0.70:
        return True, "centroid", float(sim)

    return False, "none", float(max(zs_score, sim))
# ============================================================
# UMLS MATCH HELPERS
# ============================================================
def umls_exact_match(text_clean: str):
    """Exact string match inside UMLS diabetes subset."""
    if umls_df.empty:
        return None

    mask = umls_df["STR"].str.strip().str.lower() == text_clean
    if not mask.any():
        return None

    row = umls_df[mask].iloc[0]
    return {
        "term": row["STR"],
        "cui": row["CUI"],
        "similarity": 1.0,
        "mode": "exact",
    }


def umls_embedding_match(text_clean: str):
    """SapBERT similarity match against UMLS diabetes terms."""
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
# MAIN MATCHING PIPELINE (SNOMED + UMLS + SapBERT)
# ============================================================
def analyze_user_phrase(raw_text: str, desc_df: pd.DataFrame):
    """
    Full decision pipeline.

    1) Preprocess input (Option C).
    2) If exact SNOMED synonym → 100% match.
    3) Check diabetes relevance.
    4) SNOMED SapBERT similarity.
    5) UMLS fallback (exact + similarity).
    6) Choose best candidate.
    7) Thresholds:
       - >= 0.80 → auto
       - 0.70–0.79 → review
       - < 0.70 → discard
    """

    # 0. preprocess
    text_clean = preprocess_text(raw_text)

    # 1. exact SNOMED synonym already present
    df_terms = desc_df["term"].astype(str).str.strip().str.lower()
    if text_clean in df_terms.values:
        row = desc_df[df_terms == text_clean].iloc[0]
        cid = row["conceptId"]
        term = row["term"]

        return {
            "diabetes_related": True,
            "rel_method": "exact-snomed-synonym",
            "relevance_score": 1.0,
            "existing_synonym": True,
            "decision_class": "auto",
            "snomed": {
                "matched": True,
                "decision": "Exact SNOMED synonym found",
                "concept": term,
                "conceptId": cid,
                "similarity": 1.0,
            },
            "umls": {
                "available": not umls_df.empty,
                "matched": False,
                "decision": None,
                "concept": None,
                "cui": None,
                "similarity": None,
            },
            "final": {
                "source": "SNOMED",
                "concept": term,
                "id": cid,
                "similarity": 1.0,
                "confidence": "high",
            },
        }

    # 2. diabetes relevance check
    related, rel_method, rel_score = is_diabetes_related(text_clean)
    if not related:
        return {
            "diabetes_related": False,
            "reason": f"Not diabetes-related (score={rel_score:.3f}, method={rel_method})",
        }

    # 3. SNOMED embedding match
    user_emb = embed_sapbert([text_clean])

    if term_embeddings.shape[0] == 0:
        snomed_concept = None
        snomed_id = None
        snomed_sim = 0.0
    else:
        snomed_sims = cosine_similarity(user_emb, term_embeddings)[0]
        snomed_idx = int(np.argmax(snomed_sims))
        snomed_concept = all_terms[snomed_idx]
        snomed_id = concept_ids[snomed_idx]
        snomed_sim = float(snomed_sims[snomed_idx])

    if snomed_sim >= 0.80:
        snomed_decision = "High match — SNOMED concept auto-accepted (≥ 0.80)"
    elif snomed_sim >= 0.70:
        snomed_decision = "Borderline SNOMED match (0.70–0.79)"
    else:
        snomed_decision = "Low SNOMED similarity (< 0.70)"

    # 4. UMLS (exact, then embedding)
    umls_info = None
    umls_decision = None
    umls_sim_val = -1.0

    exact = umls_exact_match(text_clean)
    if exact is not None:
        umls_info = exact
        umls_decision = "Exact UMLS diabetes concept"
        umls_sim_val = 1.0
    else:
        emb_match = umls_embedding_match(text_clean)
        if emb_match is not None:
            umls_info = emb_match
            umls_sim_val = emb_match["similarity"]
            if umls_sim_val >= 0.80:
                umls_decision = "High match — UMLS diabetes concept (≥ 0.80)"
            elif umls_sim_val >= 0.70:
                umls_decision = "Borderline UMLS match (0.70–0.79)"
            else:
                umls_decision = "Low UMLS similarity (< 0.70)"

    # 5. choose best candidate across SNOMED + UMLS
    best_source = None
    best_concept = None
    best_id = None
    best_sim = 0.0
    best_conf = "low"

    if snomed_sim >= umls_sim_val:
        best_source = "SNOMED"
        best_concept = snomed_concept
        best_id = snomed_id
        best_sim = snomed_sim
    else:
        if umls_info is not None:
            best_source = "UMLS"
            best_concept = umls_info["term"]
            best_id = umls_info["cui"]
            best_sim = umls_info["similarity"]

    # 6. decision class based on best_sim
    if best_source is None or best_concept is None:
        decision_class = "discard"
        best_sim = 0.0
        best_conf = "low"
    else:
        if best_sim >= 0.80:
            decision_class = "auto"
            best_conf = "high"
        elif best_sim >= 0.70:
            decision_class = "review"
            best_conf = "medium"
        else:
            decision_class = "discard"
            best_conf = "low"

    return {
        "diabetes_related": True,
        "rel_method": rel_method,
        "relevance_score": rel_score,
        "existing_synonym": False,
        "decision_class": decision_class,
        "snomed": {
            "matched": snomed_concept is not None,
            "decision": snomed_decision,
            "concept": snomed_concept,
            "conceptId": snomed_id,
            "similarity": snomed_sim,
        },
        "umls": {
            "available": not umls_df.empty,
            "matched": umls_info is not None,
            "decision": umls_decision,
            "concept": umls_info["term"] if umls_info else None,
            "cui": umls_info["cui"] if umls_info else None,
            "similarity": umls_sim_val if umls_info else None,
        },
        "final": {
            "source": best_source,
            "concept": best_concept,
            "id": best_id,
            "similarity": best_sim,
            "confidence": best_conf,
        },
    }


# ============================================================
# LOGGING SYSTEM
# ============================================================
def init_new_terms_log():
    if not os.path.exists(NEW_TERMS_LOG):
        with open(NEW_TERMS_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "raw_phrase",
                    "matched_term",
                    "concept_id",
                    "similarity",
                    "action",
                    "reviewer",
                    "review_time",
                ]
            )


def log_new_term(raw, term, cid, sim, action):
    init_new_terms_log()

    df = pd.read_csv(NEW_TERMS_LOG)
    if raw.lower() in df["raw_phrase"].astype(str).str.lower().values:
        return

    with open(NEW_TERMS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                raw,
                term,
                cid,
                f"{sim:.3f}",
                action,
                "",
                "",
            ]
        )


# ============================================================
# ADD SYNONYM TO SNOMED RF2 (Parent Mapping)
# ============================================================
def append_new_phrase_to_subset(raw_phrase, parent_concept, parent_concept_id):
    """
    Add user phrase as a new SNOMED synonym under the chosen parent concept.
    Also update in-memory descriptions, term list and embeddings for live use.
    """
    global descriptions, all_terms, term_embeddings

    file_path = f"{BASE_DIR}/descriptions_diabetes.tsv"
    df = pd.read_csv(file_path, sep="\t", dtype=str)

    text_clean = preprocess_text(raw_phrase)
    df_terms = df["term"].astype(str).str.strip().str.lower()
    if text_clean in df_terms.values:
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
    df.to_csv(file_path, sep="\t", index=False)

    # update in-memory copies
    descriptions = pd.concat(
        [descriptions, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    all_terms.append(raw_phrase)
    new_emb = embed_sapbert([raw_phrase])
    if term_embeddings.shape[0] == 0:
        term_embeddings_new = new_emb
    else:
        term_embeddings_new = np.vstack([term_embeddings, new_emb])
    # reassign cached embeddings
    globals()["term_embeddings"] = term_embeddings_new

    return True
# ============================================================
# MAIN USER INTERFACE
# ============================================================
st.title("🩸 Hybrid Diabetes Concept Matcher (SNOMED CT + UMLS + SapBERT)")

st.write("Enter a clinical phrase. Example:")
st.code("my sugar is high today, feet burning, shaky after meals")

# USER INPUT
user_input = st.text_input("Phrase:", key="text_input")

# SUBMIT BUTTON (required, no auto-run)
run_btn = st.button("🔎 Analyze Phrase")


# ============================================================
# RUN ANALYSIS ONLY WHEN BUTTON CLICKED
# ============================================================
if run_btn and user_input.strip() != "":
    res = analyze_user_phrase(user_input, descriptions)

    if not res["diabetes_related"]:
        st.error(res["reason"])
    else:
        st.success(f"Diabetes-related (method={res['rel_method']}, score={res['relevance_score']:.3f})")

        st.markdown("## 🧬 SNOMED CT Match")

        sn = res["snomed"]
        st.write(f"**Decision:** {sn['decision']}")
        st.write(f"**Best SNOMED concept:** {sn['concept']}")
        st.write(f"**Concept ID:** `{sn['conceptId']}`")
        st.write(f"**Similarity:** `{sn['similarity']:.3f}`")

        st.markdown("## 🧬 UMLS Match (Fallback)")

        umls = res["umls"]
        if umls["available"]:
            if umls["matched"]:
                st.info(f"**Matched UMLS term:** {umls['concept']} (CUI: `{umls['cui']}`)")
                st.write(f"Similarity: `{umls['similarity']}`")
                st.write(f"Decision: {umls['decision']}")
            else:
                st.warning("No UMLS match found")
        else:
            st.warning("UMLS file not loaded")

        st.markdown("## ✅ Final Suggested Concept")

        f = res["final"]
        if f["source"] is None:
            st.error("❌ No suitable SNOMED or UMLS concept. Tag discarded (<0.70).")
        else:
            st.success(
                f"Source: **{f['source']}** — Concept: **{f['concept']}** "
                f"(`{f['id']}`), similarity: `{f['similarity']:.3f}` "
                f"({f['confidence']} confidence)"
            )

        # ====================================================
        # HANDLE ADDING SYNONYM TO SNOMED
        # ====================================================
        if res["decision_class"] == "auto":
            st.markdown("### ➕ Auto-Add to SNOMED?")
            if st.checkbox(
                f"Add this phrase as a new SNOMED synonym "
                f"(parent: {f['concept']} / {f['id']})"
            ):
                success = append_new_phrase_to_subset(user_input, f["concept"], f["id"])
                if success:
                    st.success("Added as SNOMED synonym.")
                    log_new_term(user_input, f["concept"], f["id"], f["similarity"], "added")
                else:
                    st.info("This term already exists in the SNOMED subset.")

        elif res["decision_class"] == "review":
            st.warning("🔎 Borderline similarity (0.70–0.79). Requires admin review.")

            if st.checkbox("Queue this phrase for review?"):
                key = user_input.lower().strip()
                if key not in st.session_state["queued_phrases"]:
                    log_new_term(
                        user_input,
                        f["concept"],
                        f["id"],
                        f["similarity"],
                        "queued"
                    )
                    st.session_state["queued_phrases"].add(key)
                    st.success("Queued for admin review.")
                else:
                    st.info("Already queued.")

        else:
            st.error("❌ Similarity < 0.70 — discarded as non-actionable.")


# ============================================================
# ADMIN REVIEW SECTION
# ============================================================
st.markdown("## 🔐 Admin Review (Approve / Reject Queued Phrases)")

with st.expander("Admin Review Panel"):
    if os.path.exists(NEW_TERMS_LOG):
        df = pd.read_csv(NEW_TERMS_LOG)

        # Remove duplicates (keep latest)
        df["lower"] = df["raw_phrase"].str.lower()
        df = df.sort_values("timestamp")
        df = df[~df["lower"].duplicated(keep="last")].drop(columns=["lower"])
        df.to_csv(NEW_TERMS_LOG, index=False)

        pending = df[df["action"] == "queued"]

        if pending.empty:
            st.info("No queued items.")
        else:
            pending = pending.reset_index(drop=True)

            for i, row in pending.iterrows():
                phrase = row["raw_phrase"]
                term = row["matched_term"]
                cid = row["concept_id"]
                score = row["similarity"]

                st.write(f"### Phrase: {phrase}")
                st.write(f"Suggested Parent Concept: **{term}** (`{cid}`)")
                st.write(f"Similarity: `{score}`")

                c1, c2 = st.columns(2)

                if c1.button("Approve", key=f"approve_{i}"):
                    append_new_phrase_to_subset(phrase, term, cid)
                    df.loc[i, "action"] = "approved"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec='seconds')
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.success("Approved & added to SNOMED.")
                    st.rerun()

                if c2.button("Reject", key=f"reject_{i}"):
                    df.loc[i, "action"] = "rejected"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec='seconds')
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.warning("Rejected.")
                    st.rerun()


# ============================================================
# KNOWLEDGE GRAPH VISUALIZATION
# ============================================================
st.markdown("## 📡 Knowledge Graph (SNOMED Hierarchy)")

desc_df = pd.read_csv(f"{BASE_DIR}/descriptions_diabetes.tsv", sep="\t", dtype=str)
rel_df = pd.read_csv(f"{BASE_DIR}/relationships_diabetes.tsv", sep="\t", dtype=str)

desc_df = desc_df[desc_df["active"] == "1"]
rel_df = rel_df[rel_df["active"] == "1"]

G = nx.DiGraph()
for cid in desc_df["conceptId"].unique():
    G.add_node(cid)

isa_edges = rel_df[rel_df["typeId"] == "116680003"]
for _, row in isa_edges.iterrows():
    parent = row["destinationId"]
    child = row["sourceId"]
    if parent in G.nodes and child in G.nodes:
        G.add_edge(parent, child)

new_synonyms = desc_df[desc_df["id"].str.contains("user_", na=False)]

with st.expander("Show Knowledge Graph"):
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightblue", ax=ax)

    new_parents = new_synonyms["conceptId"].unique()
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=new_parents,
        node_color="orange",
        node_size=180,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3, ax=ax)
    ax.set_title("Knowledge Graph\n(Orange = Concepts with New Synonyms)")
    ax.axis("off")

    st.pyplot(fig)

    st.info(f"New synonyms added: **{len(new_synonyms)}**")


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================
st.markdown("## 📈 Analytics")

with st.expander("View Analytics"):
    if os.path.exists(NEW_TERMS_LOG):
        log_df = pd.read_csv(NEW_TERMS_LOG)

        total_added = (log_df["action"] == "added").sum()
        total_queued = (log_df["action"] == "queued").sum()
        total_approved = (log_df["action"] == "approved").sum()
        total_rejected = (log_df["action"] == "rejected").sum()

        st.metric("Auto-added Synonyms", total_added)
        st.metric("Queued Items", total_queued)
        st.metric("Approved Items", total_approved)
        st.metric("Rejected Items", total_rejected)

        if not log_df.empty:
            log_df["day"] = pd.to_datetime(log_df["timestamp"]).dt.date
            trend = log_df.groupby("day").size()

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(trend.index, trend.values)
            ax2.set_title("Daily Activity Trend")
            ax2.set_ylabel("Actions")
            ax2.set_xlabel("Date")
            st.pyplot(fig2)
    else:
        st.info("No analytics available yet.")


# ============================================================
# ZIP DOWNLOAD
# ============================================================
st.markdown("## 📦 Download Updated SNOMED Subset")

def make_zip():
    zip_path = f"{BASE_DIR}/subset_updated.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [
            "descriptions_diabetes.tsv",
            "concepts_diabetes.tsv",
            "relationships_diabetes.tsv",
        ]:
            z.write(f"{BASE_DIR}/{f}", arcname=f)
    return zip_path

if st.button("Create ZIP"):
    zp = make_zip()
    with open(zp, "rb") as f:
        st.download_button("Download ZIP", f, "subset_updated.zip")


# ============================================================
# PUSH TO GITHUB
# ============================================================
st.markdown("## 🚀 Push Updated TSV to GitHub")

with st.expander("GitHub Sync Settings"):
    repo = st.text_input("Repository (e.g., username/diabetes-matcher-ui)")
    token = st.text_input("GitHub Token", type="password")
    file_to_push = f"{BASE_DIR}/descriptions_diabetes.tsv"

    if st.button("Push to GitHub"):
        if not repo or not token:
            st.error("Missing repo or token.")
        else:
            api_url = f"https://api.github.com/repos/{repo}/contents/diabetes_subset_rf2/descriptions_diabetes.tsv"

            with open(file_to_push, "rb") as f:
                content = base64.b64encode(f.read()).decode()

            headers = {"Authorization": f"token {token}"}
            resp = requests.get(api_url, headers=headers)

            sha = resp.json().get("sha") if resp.status_code == 200 else None

            payload = {
                "message": "Auto-update from Streamlit app",
                "content": content,
                "branch": "main",
            }

            if sha:
                payload["sha"] = sha

            r = requests.put(api_url, headers=headers, data=json.dumps(payload))

            if r.status_code in [200, 201]:
                st.success("Pushed to GitHub successfully!")
            else:
                st.error(f"Error: {r.text}")
