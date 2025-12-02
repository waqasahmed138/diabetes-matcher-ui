import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import csv
import base64
import requests
import json
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
def is_diabetes_related(text):
    t = text.lower().strip()

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
def umls_exact_match(text_clean):
    if umls_df.empty:
        return None

    mask = umls_df["STR"].str.lower() == text_clean
    if not mask.any():
        return None

    row = umls_df[mask].iloc[0]
    return {
        "term": row["STR"],
        "cui": row["CUI"],
        "similarity": 1.0,
        "mode": "exact",
    }


def umls_embedding_match(text_clean):
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
# MAIN MATCHING PIPELINE (Option C)
# ============================================================
def analyze_user_phrase(text, desc_df):
    text_clean = text.lower().strip()

    # STEP 0 — exact synonym already in SNOMED subset
    existing = desc_df[desc_df["term"].str.lower() == text_clean]
    if not existing.empty:
        cid = existing.iloc[0]["conceptId"]
        term = existing.iloc[0]["term"]
        return {
            "diabetes_related": True,
            "rel_method": "existing synonym",
            "relevance_score": 1.0,
            "existing_synonym": True,
            "snomed": {
                "matched": True,
                "decision": "Exact match — existing SNOMED synonym",
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

    # STEP 1 — diabetes relevance filter
    related, rel_method, rel_score = is_diabetes_related(text_clean)
    if not related:
        return {
            "diabetes_related": False,
            "reason": f"Not diabetes-related (score={rel_score:.3f}, method={rel_method})",
        }

    # STEP 2 — SNOMED embedding match
    user_emb = embed_sapbert([text_clean])
    snomed_sims = cosine_similarity(user_emb, term_embeddings)[0]

    if snomed_sims.size == 0:
        snomed_idx = None
        snomed_concept = None
        snomed_id = None
        snomed_sim = 0.0
    else:
        snomed_idx = int(np.argmax(snomed_sims))
        snomed_concept = all_terms[snomed_idx]
        snomed_id = concept_ids[snomed_idx]
        snomed_sim = float(snomed_sims[snomed_idx])

    if snomed_sim >= 0.85:
        snomed_decision = "High match — existing SNOMED concept recognized"
        snomed_matched = True
        snomed_conf = "high"
    elif snomed_sim >= 0.60:
        snomed_decision = "Medium match — possible child SNOMED concept"
        snomed_matched = True
        snomed_conf = "medium"
    else:
        snomed_decision = (
            "Low match — diabetes-related but NO suitable SNOMED concept found"
        )
        snomed_matched = False
        snomed_conf = "low"

    # If SNOMED matched with at least medium confidence
    if snomed_matched:
        return {
            "diabetes_related": True,
            "rel_method": rel_method,
            "relevance_score": rel_score,
            "existing_synonym": False,
            "snomed": {
                "matched": True,
                "decision": snomed_decision,
                "concept": snomed_concept,
                "conceptId": snomed_id,
                "similarity": snomed_sim,
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
                "concept": snomed_concept,
                "id": snomed_id,
                "similarity": snomed_sim,
                "confidence": snomed_conf,
            },
        }

    # STEP 3 — SNOMED low → try UMLS exact match
    umls_info = None
    umls_matched = False
    umls_decision = None
    umls_conf = "low"

    exact = umls_exact_match(text_clean)
    if exact is not None:
        umls_info = exact
        umls_matched = True
        umls_decision = (
            "High match — existing UMLS diabetes concept (exact string match)"
        )
        umls_conf = "high"
    else:
        # STEP 4 — UMLS embedding match
        emb_match = umls_embedding_match(text_clean)
        if emb_match is not None:
            umls_info = emb_match
            if umls_info["similarity"] >= 0.85:
                umls_matched = True
                umls_decision = (
                    "High match — UMLS diabetes concept (embedding similarity)"
                )
                umls_conf = "high"
            elif umls_info["similarity"] >= 0.60:
                umls_matched = True
                umls_decision = (
                    "Medium match — UMLS diabetes concept (embedding similarity)"
                )
                umls_conf = "medium"
            else:
                umls_matched = False
                umls_decision = "Low match — no strong UMLS diabetes concept found"

    # STEP 5 — choose final match
    if umls_matched and umls_info is not None:
        return {
            "diabetes_related": True,
            "rel_method": rel_method,
            "relevance_score": rel_score,
            "existing_synonym": False,
            "snomed": {
                "matched": False,
                "decision": snomed_decision,
                "concept": snomed_concept,
                "conceptId": snomed_id,
                "similarity": snomed_sim,
            },
            "umls": {
                "available": not umls_df.empty,
                "matched": True,
                "decision": umls_decision,
                "concept": umls_info["term"],
                "cui": umls_info["cui"],
                "similarity": umls_info["similarity"],
            },
            "final": {
                "source": "UMLS",
                "concept": umls_info["term"],
                "id": umls_info["cui"],
                "similarity": umls_info["similarity"],
                "confidence": umls_conf,
            },
        }

    # STEP 6 — both SNOMED and UMLS are low → pick best low-confidence
    best_source = None
    best_concept = None
    best_id = None
    best_sim = 0.0

    umls_sim_val = umls_info["similarity"] if umls_info is not None else -1.0

    if snomed_sim >= umls_sim_val:
        best_source = "SNOMED"
        best_concept = snomed_concept
        best_id = snomed_id
        best_sim = snomed_sim
    else:
        best_source = "UMLS"
        best_concept = umls_info["term"]
        best_id = umls_info["cui"]
        best_sim = umls_info["similarity"]

    return {
        "diabetes_related": True,
        "rel_method": rel_method,
        "relevance_score": rel_score,
        "existing_synonym": False,
        "snomed": {
            "matched": False,
            "decision": snomed_decision,
            "concept": snomed_concept,
            "conceptId": snomed_id,
            "similarity": snomed_sim,
        },
        "umls": {
            "available": not umls_df.empty,
            "matched": False,
            "decision": umls_decision,
            "concept": umls_info["term"] if umls_info else None,
            "cui": umls_info["cui"] if umls_info else None,
            "similarity": umls_info["similarity"] if umls_info else None,
        },
        "final": {
            "source": best_source,
            "concept": best_concept,
            "id": best_id,
            "similarity": best_sim,
            "confidence": "low",
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
# ADD SYNONYM TO SNOMED RF2  (Parent Mapping Option A)
# ============================================================
def append_new_phrase_to_subset(raw_phrase, parent_concept, parent_concept_id):
    global descriptions, all_terms, term_embeddings

    file_path = f"{BASE_DIR}/descriptions_diabetes.tsv"
    df = pd.read_csv(file_path, sep="\t", dtype=str)

    if raw_phrase.lower() in df["term"].astype(str).str.lower().values:
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

    descriptions = pd.concat(
        [descriptions, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    all_terms.append(raw_phrase)
    new_emb = embed_sapbert([raw_phrase])
    term_embeddings = np.vstack([term_embeddings, new_emb])

    return True


# ============================================================
# MAIN UI
# ============================================================
st.title("🩸 Hybrid Diabetes Concept Matcher (SNOMED CT + UMLS + SapBERT)")

user_input = st.text_input(
    "Enter a tag or phrase (e.g., 'my sugar is high today, feet burning, shaky after meals')"
)

if user_input:
    res = analyze_user_phrase(user_input, descriptions)

    if not res.get("diabetes_related", False):
        st.error(res["reason"])
    else:
        st.success(
            f"Diabetes-related "
            f"(method={res['rel_method']}, score={res['relevance_score']:.3f})"
        )

        final = res["final"]
        snomed = res["snomed"]
        umls_info = res["umls"]

        # ------------------ SNOMED SECTION ------------------
        st.write("### 🧬 SNOMED CT Match")
        st.write(f"Decision: **{snomed['decision']}**")
        if snomed["concept"] is not None:
            st.write(f"Best SNOMED concept: **{snomed['concept']}**")
            st.write(f"Concept ID: `{snomed['conceptId']}`")
            st.write(f"Similarity: `{snomed['similarity']:.3f}`")

        # ------------------ UMLS SECTION ------------------
        if umls_info["available"]:
            st.write("### 🧠 UMLS Diabetes Match (Fallback)")
            if umls_info["concept"] is not None:
                st.write(f"UMLS decision: **{umls_info['decision']}**")
                st.write(
                    f"Best UMLS concept: **{umls_info['concept']}** "
                    f"(CUI: `{umls_info['cui']}`)"
                )
                st.write(f"Similarity: `{umls_info['similarity']:.3f}`")
            else:
                st.info("No strong UMLS candidate found.")

        # ------------------ FINAL CHOICE SUMMARY ------------------
        st.write("### ✅ Final Suggested Concept")
        if final["source"] is None or final["concept"] is None:
            st.warning("No suitable concept found in SNOMED CT or UMLS.")
        else:
            src_label = "SNOMED CT" if final["source"] == "SNOMED" else "UMLS"
            st.write(
                f"Source: **{src_label}** — "
                f"Concept: **{final['concept']}** "
                f"(`{final['id']}`), "
                f"similarity: `{final['similarity']:.3f}` "
                f"({final['confidence']} confidence)"
            )

        # ------------------ ADD SYNONYM TO SNOMED ------------------
        if snomed["concept"] is not None:
            if st.checkbox(
                "Add this phrase as a new SNOMED synonym "
                f"(parent: {snomed['concept']} / {snomed['conceptId']})?"
            ):
                ok = append_new_phrase_to_subset(
                    user_input,
                    snomed["concept"],
                    snomed["conceptId"],
                )
                if ok:
                    st.success("Added as synonym under nearest SNOMED parent concept.")
                    log_new_term(
                        user_input,
                        snomed["concept"],
                        snomed["conceptId"],
                        snomed["similarity"],
                        "added",
                    )
                else:
                    st.info("This term already exists in the SNOMED subset.")

        # ------------------ QUEUE FOR REVIEW ------------------
        if st.checkbox("Queue this phrase for admin review?"):
            key = user_input.lower().strip()
            if key not in st.session_state["queued_phrases"]:
                log_new_term(
                    user_input,
                    snomed["concept"],
                    snomed["conceptId"],
                    snomed["similarity"],
                    "queued",
                )
                st.session_state["queued_phrases"].add(key)
                st.success("Queued for admin review.")
            else:
                st.info("Already queued for review.")
                # ============================================================
# 🔐 ADMIN REVIEW PANEL
# ============================================================
with st.expander("🔐 Admin Review (Approve / Reject Queued Phrases)"):

    if os.path.exists(NEW_TERMS_LOG):
        df = pd.read_csv(NEW_TERMS_LOG)

        # Deduplicate by phrase
        df["lower"] = df["raw_phrase"].astype(str).str.lower()
        df = df.sort_values("timestamp")
        df = df[~df["lower"].duplicated(keep="last")]
        df = df.drop(columns=["lower"])
        df.to_csv(NEW_TERMS_LOG, index=False)

        pending = df[df["action"] == "queued"]

        if pending.empty:
            st.info("No queued items for review.")
        else:
            pending = pending.reset_index(drop=True)

            for i, row in pending.iterrows():
                phrase = row["raw_phrase"]
                term   = row["matched_term"]
                cid    = row["concept_id"]
                score  = row["similarity"]

                st.markdown(f"### 🔍 Phrase: **{phrase}**")
                st.write(f"Suggested SNOMED parent: **{term}** (`{cid}`)")
                st.write(f"Similarity Score: `{score}`")

                c1, c2 = st.columns(2)

                # Approve button
                if c1.button("Approve", key=f"approve_{i}"):
                    append_new_phrase_to_subset(phrase, term, cid)

                    df.loc[i, "action"] = "approved"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec="seconds")
                    df.to_csv(NEW_TERMS_LOG, index=False)

                    st.success("Approved & added to SNOMED subset.")
                    st.rerun()

                # Reject button
                if c2.button("Reject", key=f"reject_{i}"):
                    df.loc[i, "action"] = "rejected"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec="seconds")
                    df.to_csv(NEW_TERMS_LOG, index=False)

                    st.warning("Rejected.")
                    st.rerun()

    else:
        st.info("No logs detected yet.")


# ============================================================
# 📡 KNOWLEDGE GRAPH PANEL
# ============================================================
st.subheader("📡 Knowledge Graph & Structure Visualization")

# Read updated SNOMED subset
desc_df = pd.read_csv(f"{BASE_DIR}/descriptions_diabetes.tsv", sep="\t", dtype=str)
rel_df  = pd.read_csv(f"{BASE_DIR}/relationships_diabetes.tsv", sep="\t", dtype=str)

# Only active rows
desc_df = desc_df[desc_df["active"] == "1"]
rel_df  = rel_df[rel_df["active"] == "1"]

# Build SNOMED concept graph
G = nx.DiGraph()
for cid in desc_df["conceptId"].unique():
    G.add_node(cid)

# Add IS-A edges
isa_edges = rel_df[rel_df["typeId"] == "116680003"]
for _, row in isa_edges.iterrows():
    parent = row["destinationId"]
    child  = row["sourceId"]
    if parent in G.nodes and child in G.nodes:
        G.add_edge(parent, child)

# Identify new synonyms (their parent nodes are highlighted)
new_syn_df = desc_df[desc_df["id"].str.contains("user_", na=False)]
new_parent_nodes = new_syn_df["conceptId"].unique()


with st.expander("🧠 Knowledge Graph (Before vs After Updates)"):

    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G, seed=42, k=0.35)

    # Draw normal nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=100,
        node_color="#8ecae6",
        ax=ax
    )

    # Highlight parent nodes of user-added synonyms
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(new_parent_nodes),
        node_color="orange",
        node_size=180,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=False)

    ax.set_title("Knowledge Graph of Diabetes Subset\nOrange = Nodes with Newly Added Synonyms")
    ax.axis("off")
    st.pyplot(fig)

    st.info(f"Total new synonyms added: **{len(new_syn_df)}**")


# ============================================================
# 📈 ANALYTICS DASHBOARD
# ============================================================
with st.expander("📈 Matching Analytics & Activity Summary"):

    if os.path.exists(NEW_TERMS_LOG):
        log_df = pd.read_csv(NEW_TERMS_LOG)

        total_added    = (log_df["action"] == "added").sum()
        total_queued   = (log_df["action"] == "queued").sum()
        total_approved = (log_df["action"] == "approved").sum()
        total_rejected = (log_df["action"] == "rejected").sum()

        high_hits = (log_df["similarity"].astype(float) >= 0.85).sum()
        medium_hits = ((log_df["similarity"].astype(float) >= 0.60) &
                       (log_df["similarity"].astype(float) < 0.85)).sum()

        st.metric("Synonyms Added", total_added)
        st.metric("Queued for Review", total_queued)
        st.metric("Approved", total_approved)
        st.metric("Rejected", total_rejected)
        st.metric("High-Accuracy Matches", high_hits)
        st.metric("Medium-Accuracy Matches", medium_hits)

        # Activity trend
        if not log_df.empty:
            log_df["day"] = pd.to_datetime(log_df["timestamp"]).dt.date
            trend = log_df.groupby("day").size()

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(trend.index, trend.values)
            ax2.set_title("Daily Matching Activity")
            ax2.set_ylabel("Entries")
            ax2.set_xlabel("Date")

            st.pyplot(fig2)

    else:
        st.info("No analytics generated yet.")


# ============================================================
# 📦 DOWNLOAD UPDATED SUBSET
# ============================================================
st.subheader("📦 Download Updated SNOMED Diabetes Subset")

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
    zip_file = make_zip()
    with open(zip_file, "rb") as f:
        st.download_button("Download Updated Subset", f, "subset_updated.zip")


# ============================================================
# 🚀 PUSH UPDATED SUBSET TO GITHUB
# ============================================================
with st.expander("🚀 Push Updated SNOMED Subset to GitHub"):

    repo = st.text_input("Repository (format: username/repo)")
    token = st.text_input("GitHub Token", type="password")
    file_to_push = f"{BASE_DIR}/descriptions_diabetes.tsv"

    if st.button("Push to GitHub"):
        if not repo or not token:
            st.error("Repository and Token are required.")
        else:
            api_url = (
                f"https://api.github.com/repos/{repo}/contents/"
                "diabetes_subset_rf2/descriptions_diabetes.tsv"
            )

            with open(file_to_push, "rb") as f:
                content = base64.b64encode(f.read()).decode()

            headers = {"Authorization": f"token {token}"}
            resp = requests.get(api_url, headers=headers)
            sha = resp.json().get("sha") if resp.status_code == 200 else None

            payload = {
                "message": "Auto-update via Streamlit App",
                "content": content,
                "branch": "main",
            }

            if sha:
                payload["sha"] = sha

            r = requests.put(api_url, headers=headers, data=json.dumps(payload))

            if r.status_code in [200, 201]:
                st.success("Successfully pushed to GitHub.")
            else:
                st.error(f"GitHub Error: {r.text}")
