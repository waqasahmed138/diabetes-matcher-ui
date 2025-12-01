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


# ============================================================
# SESSION STATE (prevent duplicate queue entries)
# ============================================================
if "queued_phrases" not in st.session_state:
    st.session_state["queued_phrases"] = set()


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    zero_shot = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli")
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
    inputs = sap_tokenizer(texts, padding=True,
                           truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = sap_model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy()


# ============================================================
# LOAD RF2 SUBSET
# ============================================================
@st.cache_data
def load_rf2(base_dir):
    desc = pd.read_csv(f"{base_dir}/descriptions_diabetes.tsv", sep="\t", dtype=str)
    con = pd.read_csv(f"{base_dir}/concepts_diabetes.tsv", sep="\t", dtype=str)
    rel = pd.read_csv(f"{base_dir}/relationships_diabetes.tsv", sep="\t", dtype=str)

    desc = desc[desc["active"] == "1"]
    terms = desc["term"].dropna().astype(str).unique().tolist()
    ids = desc["conceptId"].astype(str).tolist()

    return desc, con, rel, terms, ids


descriptions, concepts, relationships, all_terms, concept_ids = load_rf2(BASE_DIR)


# ============================================================
# PRECOMPUTE TERM EMBEDDINGS
# ============================================================
@st.cache_resource
def get_term_embeddings():
    return embed_sapbert(all_terms)


term_embeddings = get_term_embeddings()


# ============================================================
# DIABETES CENTROID VECTOR
# ============================================================
diabetes_anchors = [
    "diabetes",
    "diabetes mellitus",
    "type 2 diabetes",
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
# HYBRID DIABETES DETECTOR
# ============================================================
def is_diabetes_related(text):
    t = text.lower().strip()

    # zero-shot
    zs = zero_shot(t, ["diabetes", "not_related"])
    zs_label = zs["labels"][0]
    zs_score = zs["scores"][0]

    if zs_label == "diabetes" and zs_score >= 0.70:
        return True, "zero-shot", zs_score

    # centroid similarity
    emb = embed_sapbert([t])
    sim = cosine_similarity(emb, diabetes_centroid)[0][0]

    if sim >= 0.70:
        return True, "centroid", float(sim)

    return False, "none", float(max(zs_score, sim))


# ============================================================
# MATCHING PIPELINE
# ============================================================
def analyze_user_phrase(text):
    text_clean = text.lower().strip()

    related, method, rel_score = is_diabetes_related(text_clean)
    if not related:
        return {
            "diabetes_related": False,
            "reason": f"Not diabetes-related (score={rel_score:.3f}, method={method})"
        }

    user_emb = embed_sapbert([text_clean])
    sims = cosine_similarity(user_emb, term_embeddings)[0]

    idx = int(np.argmax(sims))
    match_term = all_terms[idx]
    match_id = concept_ids[idx]
    sim_score = float(sims[idx])

    if sim_score >= 0.85:
        decision = "High match — existing SNOMED concept recognized"
        matched = True
    elif sim_score >= 0.60:
        decision = "Medium match — possible child concept"
        matched = True
    else:
        decision = "Low match — diabetes-related but NO suitable SNOMED concept found"
        matched = False

    return {
        "diabetes_related": True,
        "method": method,
        "relevance_score": rel_score,
        "matched": matched,
        "decision": decision,
        "concept": match_term if matched else None,
        "conceptId": match_id if matched else None,
        "similarity": sim_score,
    }


# ============================================================
# LOGGING SYSTEM
# ============================================================
def init_new_terms_log():
    if not os.path.exists(NEW_TERMS_LOG):
        with open(NEW_TERMS_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "raw_phrase", "matched_term",
                "concept_id", "similarity", "action",
                "reviewer", "review_time"
            ])


def log_new_term(raw, term, cid, sim, action):
    init_new_terms_log()

    df = pd.read_csv(NEW_TERMS_LOG)
    if raw.lower() in df["raw_phrase"].astype(str).str.lower().values:
        return

    with open(NEW_TERMS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            raw, term, cid, f"{sim:.3f}",
            action, "", ""
        ])


# ============================================================
# ADD SYNONYM TO RF2
# ============================================================
def append_new_phrase_to_subset(raw_phrase, matched_term, matched_id):
    df = pd.read_csv(f"{BASE_DIR}/descriptions_diabetes.tsv", sep="\t", dtype=str)

    if raw_phrase.lower() in df["term"].astype(str).str.lower().values:
        return False

    new_row = {
        "id": f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "effectiveTime": datetime.now().strftime("%Y%m%d"),
        "active": "1",
        "moduleId": "999999999999999999",
        "conceptId": matched_id,
        "languageCode": "en",
        "typeId": "900000000000003001",
        "term": raw_phrase,
        "caseSignificanceId": "900000000000020002"
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(f"{BASE_DIR}/descriptions_diabetes.tsv", sep="\t", index=False)
    return True


# ============================================================
# MAIN UI
# ============================================================
st.title("🩸 Hybrid Diabetes Concept Matcher (AI + SNOMED CT)")

user_input = st.text_input("Enter a tag or phrase (e.g., 'my sugar is high today')")

if user_input:
    res = analyze_user_phrase(user_input)

    if not res["diabetes_related"]:
        st.error(res["reason"])
    else:
        st.success(f"Diabetes-related (method={res['method']}, score={res['relevance_score']:.3f})")

        st.write("### SNOMED Match Result:")
        st.write(f"Decision: **{res['decision']}**")

        if res["matched"]:
            st.write(f"Concept: **{res['concept']}**")
            st.write(f"Concept ID: `{res['conceptId']}`")
            st.write(f"Similarity: `{res['similarity']:.3f}`")

            if st.checkbox("Add this phrase as a new synonym?"):
                success = append_new_phrase_to_subset(user_input, res["concept"], res["conceptId"])
                if success:
                    st.success("Added as synonym under correct SNOMED concept.")
                    log_new_term(user_input, res["concept"], res["conceptId"], res["similarity"], "added")

        else:
            st.warning("No suitable SNOMED concept found.")

            if st.checkbox("Queue this phrase for review?"):
                key = user_input.lower().strip()
                if key not in st.session_state["queued_phrases"]:
                    log_new_term(
                        user_input,
                        res["concept"],
                        res["conceptId"],
                        res["similarity"],
                        "queued"
                    )
                    st.session_state["queued_phrases"].add(key)
                    st.success("Queued for admin review.")
                else:
                    st.info("Already queued.")


# ============================================================
# ADMIN REVIEW
# ============================================================
with st.expander("🔐 Admin Review"):
    if os.path.exists(NEW_TERMS_LOG):
        df = pd.read_csv(NEW_TERMS_LOG)

        # cleanup duplicates
        df["lower"] = df["raw_phrase"].str.lower()
        df = df.sort_values("timestamp")
        df = df[~df["lower"].duplicated(keep="last")]
        df = df.drop(columns=["lower"])
        df.to_csv(NEW_TERMS_LOG, index=False)

        pending = df[df["action"] == "queued"]

        if pending.empty:
            st.info("No queued items.")
        else:
            pending = pending.reset_index(drop=True)

            for i, row in pending.iterrows():
                phrase = row["raw_phrase"]
                term   = row["matched_term"]
                cid    = row["concept_id"]
                score  = row["similarity"]

                st.write(f"### Phrase: {phrase}")
                st.write(f"Matched Concept: **{term}** (`{cid}`)")
                st.write(f"Similarity: `{score}`")

                c1, c2 = st.columns(2)

                if c1.button("Approve", key=f"approve_{i}"):
                    append_new_phrase_to_subset(phrase, term, cid)
                    df.loc[i, "action"] = "approved"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec='seconds')
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.rerun()

                if c2.button("Reject", key=f"reject_{i}"):
                    df.loc[i, "action"] = "rejected"
                    df.loc[i, "review_time"] = datetime.now().isoformat(timespec='seconds')
                    df.to_csv(NEW_TERMS_LOG, index=False)
                    st.rerun()


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================
st.subheader("📡 Knowledge Graph & Analytics Dashboard")

desc_df = pd.read_csv(f"{BASE_DIR}/descriptions_diabetes.tsv", sep="\t", dtype=str)
rel_df  = pd.read_csv(f"{BASE_DIR}/relationships_diabetes.tsv", sep="\t", dtype=str)

desc_df = desc_df[desc_df["active"] == "1"]
rel_df  = rel_df[rel_df["active"] == "1"]

G = nx.DiGraph()
for cid in desc_df["conceptId"].unique():
    G.add_node(cid)

isa_edges = rel_df[rel_df["typeId"] == "116680003"]  # IS-A
for _, row in isa_edges.iterrows():
    parent = row["destinationId"]
    child  = row["sourceId"]
    if parent in G.nodes and child in G.nodes:
        G.add_edge(parent, child)

new_synonyms = desc_df[desc_df["id"].str.contains("user_", na=False)]

with st.expander("🧠 Knowledge Graph (Before vs After Updates)"):

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightblue", ax=ax)
    new_parents = new_synonyms["conceptId"].unique()

    nx.draw_networkx_nodes(G, pos, nodelist=new_parents,
                           node_color="orange", node_size=180, ax=ax)

    nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3, ax=ax)

    ax.set_title("Knowledge Graph\nOrange = Concepts with New Synonyms")
    ax.axis("off")

    st.pyplot(fig)

    st.info(f"New synonyms added: **{len(new_synonyms)}**")


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================
with st.expander("📈 Analytics / Stats"):

    if os.path.exists(NEW_TERMS_LOG):
        log_df = pd.read_csv(NEW_TERMS_LOG)

        total_added    = (log_df["action"] == "added").sum()
        total_queued   = (log_df["action"] == "queued").sum()
        total_approved = (log_df["action"] == "approved").sum()
        total_rejected = (log_df["action"] == "rejected").sum()

        high_hits = (log_df["similarity"].astype(float) >= 0.85).sum()
        medium_hits = ((log_df["similarity"].astype(float) >= 0.60) &
                       (log_df["similarity"].astype(float) < 0.85)).sum()

        st.metric("New Synonyms Added", total_added)
        st.metric("Queued Items", total_queued)
        st.metric("Approved Items", total_approved)
        st.metric("Rejected Items", total_rejected)
        st.metric("High-Accuracy Matches", high_hits)
        st.metric("Medium-Accuracy Matches", medium_hits)

        if not log_df.empty:
            log_df["day"] = pd.to_datetime(log_df["timestamp"]).dt.date
            trend = log_df.groupby("day").size()

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(trend.index, trend.values)
            ax2.set_title("Daily Activity Trend")
            ax2.set_ylabel("Changes")
            ax2.set_xlabel("Date")
            st.pyplot(fig2)

    else:
        st.info("No analytics yet.")


# ============================================================
# DOWNLOAD UPDATED SUBSET
# ============================================================
st.subheader("📦 Download Updated Subset")


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
with st.expander("🚀 Push Updated TSV to GitHub"):

    repo = st.text_input("Repository (e.g., username/repo)")
    token = st.text_input("GitHub Token", type="password")
    file_to_push = f"{BASE_DIR}/descriptions_diabetes.tsv"

    if st.button("Push to GitHub"):
        if not repo or not token:
            st.error("Missing repo or token")
        else:
            api_url = (
                f"https://api.github.com/repos/{repo}/contents/"
                "descriptions_diabetes.tsv"
            )

            with open(file_to_push, "rb") as f:
                content = base64.b64encode(f.read()).decode()

            headers = {"Authorization": f"token {token}"}
            resp = requests.get(api_url, headers=headers)
            sha = resp.json().get("sha") if resp.status_code == 200 else None

            payload = {
                "message": "Auto-update from Streamlit",
                "content": content,
                "branch": "main",
            }

            if sha:
                payload["sha"] = sha

            r = requests.put(api_url, headers=headers, data=json.dumps(payload))

            if r.status_code in [200, 201]:
                st.success("Pushed to GitHub!")
            else:
                st.error(f"Error: {r.text}")
