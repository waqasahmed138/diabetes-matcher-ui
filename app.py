import streamlit as st
import pandas as pd
import numpy as np
import torch, os, zipfile, requests, re, unicodedata, csv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from rapidfuzz import process, fuzz

# =============== UI CONFIG ===============
st.set_page_config(page_title="Diabetes Concept Matcher", page_icon="🩸")
st.title("🩸 Diabetes Concept Matcher (SNOMED CT subset + SapBERT)")
st.caption("Enter any tag/phrase. We’ll normalize, correct typos, and match to the closest diabetes concept.")

# =============== SETTINGS ===============
# 1) DATA SOURCE
USE_LOCAL_FOLDER = False   # set False to fetch a ZIP from GitHub
LOCAL_DIR = "./diabetes_subset_rf2"  # commit the 3 TSVs here if using local
ZIP_URL  = "https://github.com/waqasahmed138/diabetes-matcher-ui/raw/main/diabetes_subset_rf2.zip"  # RAW link if using ZIP
ZIP_PATH = "/tmp/diabetes_subset_rf2.zip"
EXTRACT_DIR = "./diabetes_subset_rf2"  # same as LOCAL_DIR intentionally

# 2) THRESHOLDS (you can expose some in the sidebar)
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Not-related threshold (cosine)", 0.0, 1.0, 0.60, 0.01)
    high_thresh = st.slider("High match threshold", 0.0, 1.0, 0.85, 0.01)
    fuzzy_score = st.slider("RapidFuzz token similarity cutoff (0-100)", 0, 100, 75, 1)
    show_topk = st.slider("Show Top-K similar concepts", 1, 10, 5, 1)
    st.markdown("---")
    st.caption("Tip: If your TSVs aren’t in the repo, set USE_LOCAL_FOLDER=False and make sure ZIP_URL points to a RAW zip file.")

# =============== DATA LOADING ===============
@st.cache_data(show_spinner=False)
def ensure_data(local_ok=True):
    if local_ok and os.path.isdir(LOCAL_DIR):
        return LOCAL_DIR
    # fallback: download & extract ZIP
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    r = requests.get(ZIP_URL, timeout=60)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

@st.cache_data(show_spinner=False)
def load_rf2(base_dir):
    desc = pd.read_csv(os.path.join(base_dir, "descriptions_diabetes.tsv"), sep="\t", dtype=str)
    con  = pd.read_csv(os.path.join(base_dir, "concepts_diabetes.tsv"), sep="\t", dtype=str)
    rel  = pd.read_csv(os.path.join(base_dir, "relationships_diabetes.tsv"), sep="\t", dtype=str)
    desc = desc[desc["active"] == "1"].copy()
    return desc, con, rel

base_dir = ensure_data(USE_LOCAL_FOLDER)
descriptions, concepts, relationships = load_rf2(base_dir)

# =============== MODEL LOADING ===============
@st.cache_resource(show_spinner=True)
def load_model():
    name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
st.success("SapBERT loaded and RF2 subset ready.", icon="✅")

# =============== BUILD LEXICON & EMBEDDINGS ===============
def normalize_unicode_lower(text): return unicodedata.normalize("NFKC", text).lower()

def strip_noise(text):
    text = re.sub(r'(https?://\S+)|(\w+@\w+\.\w+)', ' ', text)
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'[^a-z0-9\s/+-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

STOPWORDS = {
  "a","an","the","and","or","but","if","then","else","so","to","of","in","on","for","with",
  "is","are","am","be","was","were","been","being","it","this","that","these","those",
  "i","you","he","she","we","they","me","him","her","us","them","my","your","our",
  "at","by","as","from","about","how","what","why","when","where","which","who","whom"
}
ACRONYM_MAP = {
  "dm":"diabetes mellitus","t1dm":"type 1 diabetes mellitus","t2dm":"type 2 diabetes mellitus",
  "fsbs":"fingerstick blood sugar"
}
def expand_acronyms(t): return " ".join(ACRONYM_MAP.get(w, w) for w in t.split())
def normalize_numbers_units(t):
    t = re.sub(r'\bmg\s*/\s*dl\b', 'mg/dl', t)
    t = re.sub(r'\bmmol\s*/\s*l\b', 'mmol/l', t)
    return t

@st.cache_data(show_spinner=False)
def build_lexicon(desc_df):
    lex = set()
    for term in desc_df["term"].dropna().astype(str):
        t = strip_noise(normalize_unicode_lower(term))
        for tok in t.split():
            if len(tok) >= 3:
                lex.add(tok)
    lex.update({"dm","t1dm","t2dm","hba1c","glucose","insulin","hypoglycemia","hyperglycemia"})
    return lex

diabetes_lexicon = build_lexicon(descriptions)

def correct_spelling(tok, vocab, cutoff=75):
    if tok in vocab: 
        return tok
    match = process.extractOne(tok, vocab, scorer=fuzz.QRatio, score_cutoff=cutoff)
    return match[0] if match else tok

def fuzzy_in_vocab(tok, vocab, cutoff=75):
    if tok in vocab: return True
    return process.extractOne(tok, vocab, scorer=fuzz.QRatio, score_cutoff=cutoff) is not None

def clean_and_filter(text, vocab, cutoff=75, autocorrect=True):
    t = normalize_unicode_lower(text)
    t = strip_noise(t)
    t = expand_acronyms(t)
    t = normalize_numbers_units(t)
    toks = [w for w in t.split() if w not in STOPWORDS]
    kept = []
    for tok in toks:
        if len(tok) < 1: 
            continue
        if autocorrect:
            tok = correct_spelling(tok, vocab, cutoff=cutoff)
        if fuzzy_in_vocab(tok, vocab, cutoff=cutoff):
            kept.append(tok)
    return " ".join(kept)

@st.cache_resource(show_spinner=True)
def embed_terms(terms):
    vecs = []
    bs = 16
    for i in range(0, len(terms), bs):
        batch = terms[i:i+bs]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:,0,:]
        vecs.append(cls)
    return torch.cat(vecs).cpu().numpy()

all_terms = descriptions["term"].dropna().astype(str).unique().tolist()
concept_ids = descriptions.loc[descriptions["term"].notna(), "conceptId"].astype(str).tolist()
term_embeddings = embed_terms(all_terms)

# =============== MATCHING HELPERS ===============
def get_topk(input_text, k=5):
    inputs = tokenizer([input_text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    u = outputs.last_hidden_state[:,0,:].cpu().numpy()
    sims = cosine_similarity(u, term_embeddings)[0]
    idxs = np.argsort(-sims)[:k]
    rows = []
    for i in idxs:
        rows.append({
            "rank": len(rows)+1,
            "term": all_terms[i],
            "conceptId": concept_ids[i],
            "similarity": float(sims[i])
        })
    return rows

LOG_FILE = "user_tag_log.csv"
def log_entry(raw, normalized, best, best_id, score, decision):
    header = ["timestamp","raw_input","normalized","matched_term","concept_id","similarity_score","decision"]
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists: w.writerow(header)
        w.writerow([datetime.now().isoformat(timespec='seconds'), raw, normalized, best, best_id,
                    f"{score:.3f}" if score is not None else "", decision])

# =============== ONE-FORM UI ===============
tag = st.text_input("💬 Enter a tag or phrase (e.g., 'type 2 sugar level high')", "")
go = st.button("Match concept")

if go and tag.strip():
    # 1) clean + gate
    normalized = clean_and_filter(tag, diabetes_lexicon, cutoff=fuzzy_score, autocorrect=True)
    if not normalized:
        st.error("❌ Not diabetes-related (no domain tokens after cleaning).")
        log_entry(tag, "", "", "", 0.0, "Not related")
    else:
        # 2) similarity
        topk = get_topk(normalized, k=show_topk)
        best = topk[0]
        st.subheader("🔍 Result")
        st.write(f"**User Tag:** {tag}")
        st.write(f"**Normalized:** `{normalized}`")
        st.write(f"**Closest Match:** {best['term']}  \n**Concept ID:** `{best['conceptId']}`  \n**Cosine Similarity:** `{best['similarity']:.3f}`")

        # 3) decision
        if best["similarity"] < threshold:
            decision = "Not related (below threshold)"
            st.error("🔴 Below threshold — treat as NOT related to Diabetes SNOMED CT subset.")
        elif best["similarity"] >= high_thresh:
            decision = "Existing concept recognized"
            st.success("✅ High similarity — existing concept recognized.")
        else:
            decision = "Child concept candidate"
            st.warning("🟡 Medium similarity — potential child concept candidate.")

        # 4) show top-K table
        st.markdown("**Top matches**")
        df = pd.DataFrame(topk)
        st.dataframe(df, hide_index=True)

        # 5) log
        log_entry(tag, normalized, best["term"], best["conceptId"], best["similarity"], decision)

# =============== UTILITIES ===============
with st.expander("📥 Download logs"):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download user_tag_log.csv", f, file_name="user_tag_log.csv")
    else:
        st.caption("No logs yet.")
