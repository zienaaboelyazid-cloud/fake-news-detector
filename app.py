import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');
:root {
    --bg:     #0D0C14;
    --bg2:    #1C1A2E;
    --card:   #2E2C42;
    --card2:  #241F38;
    --purple: #7C6FF7;
    --purp2:  #C97BB2;
    --glow-violet: #7C6FF7;
    --glow-rose:   #C97BB2;
    --glow-teal:   #5EC4BF;
    --text:   #F4F2FF;
    --muted:  #9490B8;
    --surface: rgba(255,255,255,0.035);
    --surface-hover: rgba(255,255,255,0.06);
    --border:      rgba(124,111,247,0.18);
    --border-soft: rgba(255,255,255,0.07);
}
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: 'Cairo', sans-serif !important;
    color: var(--text) !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stMainBlockContainer"] {
    max-width: 1140px !important;
    padding: 0 2rem 5rem !important;
}
.topbar-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    background: var(--bg);
    position: sticky;
    top: 0;
    z-index: 100;
}
.logo {
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.25rem; font-weight: 900; color: var(--yellow) !important;
    min-width: 180px;
}
.logo-box {
    width: 36px; height: 36px; background: var(--yellow);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 1rem;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    padding: .45rem 1.1rem !important;
    transition: all .15s !important;
    box-shadow: none !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    color: var(--text) !important;
    background: var(--card) !important;
    transform: none !important;
}
.lang-pill-btn button,
.lang-pill-btn-active button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: .72rem !important;
    font-weight: 700 !important;
    padding: .28rem .45rem !important;
    border-radius: 8px !important;
    font-family: 'Cairo', sans-serif !important;
    transition: all .15s !important;
    box-shadow: none !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
}
.lang-pill-btn button:hover {
    border-color: var(--purple) !important;
    color: var(--text) !important;
    transform: none !important;
}
.lang-pill-btn-active button {
    background: rgba(245,197,24,.12) !important;
    border-color: var(--yellow) !important;
    color: var(--yellow) !important;
}
.hero { text-align: center; padding: 3.5rem 1rem 2.5rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(245,197,24,.1); border: 1px solid rgba(245,197,24,.3);
    color: var(--yellow); font-size: .75rem; font-weight: 700;
    letter-spacing: .06em; padding: .4rem 1.1rem;
    border-radius: 100px; margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2rem,5vw,3.5rem); font-weight: 900;
    line-height: 1.1; color: #fff; margin-bottom: 1rem;
}
.hero-title .hl { color: var(--yellow); }
.hero-sub {
    font-size: .98rem; color: var(--muted);
    max-width: 540px; margin: 0 auto 2rem; line-height: 1.85;
}
.hero-stats {
    display: flex; justify-content: center; gap: 3.5rem;
    padding-top: 2rem; border-top: 1px solid var(--border);
}
.stat-v { font-size: 2.2rem; font-weight: 900; color: var(--yellow); line-height: 1; }
.stat-l { font-size: .72rem; color: var(--muted); margin-top: .25rem; text-transform: uppercase; letter-spacing: .06em; }
.fcards-wrap { display: flex; justify-content: center; margin: 2.5rem 0; }
.fcards { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; max-width: 860px; width: 100%; }
.fcard {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    transition: border-color .2s, transform .2s;
}
.fcard:hover { border-color: var(--purp2); transform: translateY(-3px); }
.fcard-ic {
    width: 44px; height: 44px; background: rgba(124,58,237,.25);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 1.3rem; margin: 0 auto 1rem;
}
.fcard-t { font-size: .95rem; font-weight: 700; color: #fff; margin-bottom: .4rem; }
.fcard-d { font-size: .8rem; color: var(--muted); line-height: 1.75; }
.page-h { font-size: 1.8rem; font-weight: 900; color: #fff; margin: 2rem 0 .3rem; }
.page-h .hl { color: var(--yellow); }
.page-s { font-size: .88rem; color: var(--muted); margin-bottom: 1.5rem; }
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: .95rem !important;
    padding: 1rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--purp2) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.15) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }
.analyze-btn-wrap [data-testid="stButton"] > button {
    background: var(--yellow) !important;
    color: #120824 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 900 !important;
    font-size: .95rem !important;
    padding: .65rem 2.5rem !important;
    transition: all .15s !important;
}
.analyze-btn-wrap [data-testid="stButton"] > button:hover {
    background: var(--yell2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(245,197,24,.3) !important;
}
[data-testid="stSuccess"] {
    background: rgba(34,197,94,.1) !important;
    border: 1px solid rgba(34,197,94,.3) !important;
    border-radius: 12px !important; color: #4ade80 !important;
    font-weight: 700 !important; font-family: 'Cairo', sans-serif !important;
}
[data-testid="stWarning"] {
    background: rgba(245,197,24,.08) !important;
    border: 1px solid rgba(245,197,24,.2) !important;
    border-radius: 12px !important; color: var(--yellow) !important;
    font-size: .85rem !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,.1) !important;
    border: 1px solid rgba(239,68,68,.3) !important;
    border-radius: 12px !important; color: #f87171 !important;
}
[data-testid="stInfo"] {
    background: rgba(124,58,237,.1) !important;
    border: 1px solid rgba(124,58,237,.3) !important;
    border-radius: 12px !important; color: var(--purp2) !important;
}
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important; font-size: .7rem !important;
    text-transform: uppercase !important; letter-spacing: .08em !important; font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: var(--yellow) !important; font-family: 'Cairo', sans-serif !important;
    font-size: 1.4rem !important; font-weight: 900 !important;
}
h1, h2, h3 { font-family: 'Cairo', sans-serif !important; font-weight: 900 !important; color: #fff !important; }
p, .stMarkdown p { color: var(--muted) !important; font-size: .9rem !important; }
.result-real {
    background: rgba(34,197,94,.12);
    border: 1.5px solid rgba(34,197,94,.4);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: #4ade80; font-weight: 700; font-size: 1.05rem; margin-bottom: .75rem;
}
.result-fake {
    background: rgba(239,68,68,.12);
    border: 1.5px solid rgba(239,68,68,.4);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: #f87171; font-weight: 700; font-size: 1.05rem; margin-bottom: .75rem;
}
.result-uncertain {
    background: rgba(245,197,24,.1);
    border: 1.5px solid rgba(245,197,24,.3);
    border-radius: 12px; padding: 1rem 1.25rem;
    color: var(--yellow); font-weight: 700; font-size: 1.05rem; margin-bottom: .75rem;
}
.igrid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.icard { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; }
.icard-v { font-size: 2rem; font-weight: 900; color: var(--yellow); line-height: 1; margin-bottom: .3rem; }
.icard-l { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.bar-w  { margin-bottom: .75rem; }
.bar-hd { display: flex; justify-content: space-between; margin-bottom: .35rem; }
.bar-nm { font-size: .82rem; color: var(--text); font-weight: 600; }
.bar-nu { font-size: .78rem; color: var(--muted); }
.bar-tr { height: 8px; background: rgba(124,58,237,.15); border-radius: 100px; overflow: hidden; }
.bar-fi { height: 100%; background: linear-gradient(90deg, var(--purple), var(--yellow)); border-radius: 100px; }
.hist-row {
    display: flex; align-items: center; gap: 1rem;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: .85rem 1.1rem;
    margin-bottom: .5rem; font-size: .82rem;
}
.hist-badge { font-size: .68rem; font-weight: 700; padding: .2rem .75rem; border-radius: 100px; white-space: nowrap; text-transform: uppercase; }
.hist-real { background: rgba(34,197,94,.15); color: #4ade80; }
.hist-fake { background: rgba(239,68,68,.15);  color: #f87171; }
.hist-unct { background: rgba(245,197,24,.15); color: var(--yellow); }
.hist-text { flex: 1; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hist-src  { color: var(--muted); font-size: .72rem; white-space: nowrap; }
.empty { text-align: center; padding: 3.5rem 1rem; color: var(--muted); font-size: .9rem; }
.footer {
    margin-top: 4rem; padding-top: 1.75rem; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
}
.f-logo { font-weight: 900; color: var(--yellow); font-size: 1rem; }
.f-copy { font-size: .72rem; color: var(--muted); }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--purple); border-radius: 100px; }
</style>
""", unsafe_allow_html=True)
# SESSION STATE
if "page"    not in st.session_state: st.session_state.page    = "home"
if "lang"    not in st.session_state: st.session_state.lang    = "English"
if "history" not in st.session_state: st.session_state.history = []
# LANGUAGE STRINGS
LANG = {
    "English": {
        "brand": "ShieldAI",
        "nav_home": "Home", "nav_detector": "Detector", "nav_insights": "Insights",
        "badge": "AI-Powered • Multi-language • Real-time",
        "hero_title": 'Detect <span class="hl">Fake News</span><br>Before It Spreads',
        "s_layers": "Detection Layers", "s_langs": "Languages", "s_news": "News in Dataset",
        "f1_title": "Dataset Match", "f1_desc": "Compares against a labeled real/fake dataset using TF-IDF cosine similarity with 85% threshold.",
        "f2_title": "Live Scraping",  "f2_desc": "Scrapes Youm7 headlines in real-time and checks similarity to trusted news.",
        "f3_title": "Gemini AI",      "f3_desc": "When both methods are inconclusive, Google Gemini provides deep contextual analysis.",
        "footer": "Built with Streamlit, scikit-learn & Gemini AI",
        "page_title": 'News <span class="hl">Detector</span>',
        "page_sub": "Paste any news article or headline below and click Analyze.",
        "analyze_btn": "🔍 Analyze",
        "textarea_placeholder": "📰 Paste the news here…",
        "warning_empty": "Please write a news article first.",
        "spinner": "Analyzing…",
        "result": "📊 Result",
        "closest": "📚 Youm7 news",
        "dataset_sim": "Dataset Similarity", "youm7_sim": "Youm7 Similarity", "source": "Source",
        "real": "🟢 Real News", "fake": "🔴 Fake News",
        "partial": "🟡 Partially similar to Youm7 → Uncertain",
        "similar": "🟢 Similar to Youm7 News → Likely Real",
        "dataset_warning": "⚠️ This result is based on similarity matching only.",
        "ai_warning": "⚠️ AI predictions may not always be accurate.",
        "insights_title": 'Session <span class="hl">Insights</span>',
        "insights_sub": "Stats from all analyses done this session.",
        "ins_total": "Total Analyzed", "ins_real": "Real", "ins_fake": "Fake", "ins_unct": "Uncertain",
        "ins_sim": "Avg Similarity", "ins_sources": "Sources Used", "ins_hist": "Analysis History",
        "ins_empty": "No analyses yet — go to Detector and try some news!",
    },
    "العربية": {
        "brand": "كاشف الأخبار",
        "nav_home": "الرئيسية", "nav_detector": "الكاشف", "nav_insights": "إحصائيات",
        "badge": "مدعوم بالذكاء الاصطناعي • متعدد اللغات • فوري",
        "hero_title": 'اكتشف <span class="hl">الأخبار المزيفة</span><br>قبل أن تنتشر',
        "s_layers": "طبقات التحقق", "s_langs": "لغات", "s_news": "خبر في قاعدة البيانات",
        "f1_title": "مطابقة البيانات", "f1_desc": "يقارن الخبر بقاعدة بيانات مصنفة باستخدام تشابه TF-IDF بعتبة 85%.",
        "f2_title": "تتبع مباشر",      "f2_desc": "يجلب عناوين اليوم السابع لحظياً ويقارنها بالخبر.",
        "f3_title": "Gemini AI",        "f3_desc": "عند عدم اليقين، يستخدم Google Gemini للتحليل السياقي العميق.",
        "footer": "مبني بـ Streamlit و scikit-learn و Gemini AI",
        "page_title": '<span class="hl">كاشف</span> الأخبار',
        "page_sub": "الصق أي خبر أدناه واضغط تحليل.",
        "analyze_btn": "🔍 تحليل",
        "textarea_placeholder": "📰 الصق الخبر هنا…",
        "warning_empty": "اكتب الخبر أولاً.",
        "spinner": "جاري التحليل…",
        "result": "📊 النتيجة",
        "closest": " اخبار من اليوم السابع",
        "dataset_sim": "تشابه الداتا", "youm7_sim": "تشابه اليوم السابع", "source": "المصدر",
        "real": "🟢 خبر حقيقي", "fake": "🔴 خبر مزيف",
        "partial": "🟡 مشابه جزئيًا لأخبار اليوم السابع → غير مؤكد",
        "similar": "🟢 مشابه لأخبار اليوم السابع → غالبًا حقيقي",
        "dataset_warning": "⚠️ النتيجة مبنية على التشابه فقط.",
        "ai_warning": "⚠️ توقعات الذكاء الاصطناعي قد تكون غير دقيقة.",
        "insights_title": '<span class="hl">إحصائيات</span> الجلسة',
        "insights_sub": "إحصائيات جميع التحليلات في هذه الجلسة.",
        "ins_total": "إجمالي التحليلات", "ins_real": "حقيقية", "ins_fake": "مزيفة", "ins_unct": "غير مؤكدة",
        "ins_sim": "متوسط التشابه", "ins_sources": "المصادر المستخدمة", "ins_hist": "سجل التحليلات",
        "ins_empty": "لا توجد تحليلات بعد — اذهب للكاشف وجرب!",
    },}
LANG_KEYS  = list(LANG.keys())
LANG_SHORT = {"English": "EN", "العربية": "AR"}
page = st.session_state.page
lang = st.session_state.lang
txt  = LANG[lang]
# TOPBAR
st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
logo_col, nav_col, lang_col = st.columns([2, 5, 2])
with logo_col:
    st.markdown(f'<div class="logo"><div class="logo-box">🛡️</div>{txt["brand"]}</div>', unsafe_allow_html=True)
with nav_col:
    n1, n2, n3 = st.columns(3)
    nav_pages  = [("home", txt["nav_home"]), ("detector", txt["nav_detector"]), ("insights", txt["nav_insights"])]
    for col, (p, label) in zip([n1, n2, n3], nav_pages):
        with col:
            active_marker = " ●" if page == p else ""
            if st.button(label + active_marker, key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()
with lang_col:
    lang_subcols = st.columns(len(LANG_KEYS))
    for col, l in zip(lang_subcols, LANG_KEYS):
        with col:
            active_cls = "lang-pill-btn-active" if lang == l else "lang-pill-btn"
            st.markdown(f'<div class="{active_cls}" style="display:flex;justify-content:center">', unsafe_allow_html=True)
            if st.button(LANG_SHORT[l], key=f"lang_{l}"):
                st.session_state.lang = l
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
page = st.session_state.page
lang = st.session_state.lang
txt  = LANG[lang]
# GEMINI SETUP
genai.configure(api_key="AIzaSyC5gI11WfVgW98qFndLDmOypC8rTCX2WbQ")
ai_model = genai.GenerativeModel("gemini-2.5-flash")
# LOAD DATA + MODEL
@st.cache_resource
def load_models():
    df         = pd.read_csv(r"C:\Users\scs\Downloads\harder_realistic_fake_news_dataset.csv")
    model      = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    vectors    = vectorizer.transform(df["text"].astype(str))
    return df, model, vectorizer, vectors
df, model, vectorizer, dataset_vectors = load_models()
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text
def search_dataset(news):
    cleaned_news = clean_text(news)
    news_vec     = vectorizer.transform([cleaned_news])
    sims         = cosine_similarity(news_vec, dataset_vectors)[0]
    best_index   = np.argmax(sims)
    best_score   = sims[best_index]
    matched_row  = df.iloc[best_index]
    THRESHOLD    = 0.85
    if best_score >= THRESHOLD:
        if matched_row["label"] == "real":
            label = "REAL"
        elif matched_row["label"] == "fake":
            label = "FAKE"
        return best_score, matched_row["text"], label
    return best_score, None, "UNCERTAIN"
def ask_ai(news):
    prompt = f"""
You are a fake news detector.
Analyze this news and respond ONLY:
Label: REAL / FAKE / UNCERTAIN
Reason: short explanation
News:
{news}"""
    response = ai_model.generate_content(prompt)
    return response.text
@st.cache_data
def get_youm7_news():
    url     = "https://www.youm7.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    res     = requests.get(url, headers=headers, timeout=10)
    soup    = BeautifulSoup(res.text, "html.parser")
    headlines = soup.find_all("h3")
    news = []
    for h in headlines:
        text = h.get_text().strip()
        if len(text) > 10:
            news.append(text)
    return news
youm7_news    = get_youm7_news()
youm7_vectors = vectorizer.transform([clean_text(x) for x in youm7_news])
def check_youm7(news):
    cleaned      = clean_text(news)
    vec          = vectorizer.transform([cleaned])
    sims         = cosine_similarity(vec, youm7_vectors)[0]
    best_score   = np.max(sims)
    best_idx     = np.argmax(sims)
    matched_news = youm7_news[best_idx]
    return best_score, matched_news
# HELPER
def result_class(label):
    lbl = label.upper()
    if any(k in lbl for k in ["REAL", "🟢", "حقيقي", "LIKELY"]):
        return "result-real"
    elif any(k in lbl for k in ["FAKE", "🔴", "مزيف"]):
        return "result-fake"
    else:
        return "result-uncertain"
# PAGE: HOME
if page == "home":
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">✨ {txt['badge']}</div>
        <h1 class="hero-title">{txt['hero_title']}</h1>
        <div class="hero-stats">
            <div><div class="stat-v">3</div><div class="stat-l">{txt['s_layers']}</div></div>
            <div><div class="stat-v">2</div><div class="stat-l">{txt['s_langs']}</div></div>
            <div><div class="stat-v">{len(df):,}</div><div class="stat-l">{txt['s_news']}</div></div>
        </div>
    </div>
    <div class="fcards-wrap">
        <div class="fcards">
            <div class="fcard"><div class="fcard-ic">🔍</div><div class="fcard-t">{txt['f1_title']}</div><div class="fcard-d">{txt['f1_desc']}</div></div>
            <div class="fcard"><div class="fcard-ic">📡</div><div class="fcard-t">{txt['f2_title']}</div><div class="fcard-d">{txt['f2_desc']}</div></div>
            <div class="fcard"><div class="fcard-ic">🤖</div><div class="fcard-t">{txt['f3_title']}</div><div class="fcard-d">{txt['f3_desc']}</div></div>
        </div>
    </div>
    <div class="footer">
        <div class="f-logo">🛡️ {txt['brand']}</div>
        <div class="f-copy">{txt['footer']}</div>
    </div>
    """, unsafe_allow_html=True)
# PAGE: DETECTOR
elif page == "detector":
    st.markdown(f'<div class="page-h">{txt["page_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-s">{txt["page_sub"]}</p>', unsafe_allow_html=True)
    news_input = st.text_area(
        "news",
        height=220,
        label_visibility="collapsed",
        placeholder=txt["textarea_placeholder"],)
    st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
    analyze_clicked = st.button(txt["analyze_btn"], key="analyze_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    if analyze_clicked:
        if news_input.strip():
            with st.spinner(txt["spinner"]):
                dataset_sim, dataset_match, dataset_label = search_dataset(news_input)
                youm7_sim, youm7_match                    = check_youm7(news_input)
                if dataset_label != "UNCERTAIN":
                    if dataset_label == "REAL":
                        final = txt["real"]
                    elif dataset_label == "FAKE":
                        final = txt["fake"]
                    source = "Dataset Match"
                elif youm7_sim > 0.60:
                    final  = txt["similar"]
                    source = "Youm7 Match"
                elif youm7_sim > 0.35:
                    final  = txt["partial"]
                    source = "Youm7 Partial"
                else:
                    final  = ask_ai(news_input)
                    source = "Gemini AI"
            st.session_state.history.append({
                "text":        news_input[:80] + ("…" if len(news_input) > 80 else ""),
                "label":       final,
                "source":      source,
                "dataset_sim": dataset_sim,
                "youm7_sim":   youm7_sim,})
            st.subheader(txt["result"])
            st.markdown(f'<div class="{result_class(final)}">{final}</div>', unsafe_allow_html=True)
            st.warning(txt["dataset_warning"])
            if source == "Gemini AI":
                st.warning(txt["ai_warning"])
            c1, c2, c3 = st.columns(3)
            c1.metric(txt["dataset_sim"], f"{dataset_sim:.2f}")
            c2.metric(txt["youm7_sim"],   f"{youm7_sim:.2f}")
            c3.metric(txt["source"],      source)
            st.subheader(txt["closest"])
            st.info(youm7_match)
        else:
            st.warning(txt["warning_empty"])
# PAGE: INSIGHTS
elif page == "insights":
    history = st.session_state.history
    total   = len(history)
    def has(label, keys): return any(k in label for k in keys)
    real_c  = sum(1 for x in history if has(x["label"], ["🟢", "REAL", "حقيقي", "Likely"]))
    fake_c  = sum(1 for x in history if has(x["label"], ["🔴", "FAKE", "مزيف"]))
    unct_c  = total - real_c - fake_c
    avg_ds  = round(sum(x["dataset_sim"] for x in history) / total, 2) if total else 0
    avg_y7  = round(sum(x["youm7_sim"]   for x in history) / total, 2) if total else 0
    sources = {}
    for x in history: sources[x["source"]] = sources.get(x["source"], 0) + 1
    st.markdown(f'<div class="page-h" style="margin-top:2rem">{txt["insights_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-s">{txt["insights_sub"]}</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="igrid">
        <div class="icard"><div class="icard-v">{total}</div><div class="icard-l">{txt['ins_total']}</div></div>
        <div class="icard"><div class="icard-v" style="color:#4ade80">{real_c}</div><div class="icard-l">{txt['ins_real']}</div></div>
        <div class="icard"><div class="icard-v" style="color:#f87171">{fake_c}</div><div class="icard-l">{txt['ins_fake']}</div></div>
        <div class="icard"><div class="icard-v" style="color:#f59e0b">{unct_c}</div><div class="icard-l">{txt['ins_unct']}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if total:
        ds_pct   = int(avg_ds * 100)
        y7_pct   = int(avg_y7 * 100)
        src_bars = "".join(
            f'<div class="bar-w"><div class="bar-hd"><span class="bar-nm">{s}</span><span class="bar-nu">{c}</span></div>'
            f'<div class="bar-tr"><div class="bar-fi" style="width:{int(c/total*100)}%"></div></div></div>'
            for s, c in sources.items())
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem">
            <div class="icard">
                <div class="icard-l" style="margin-bottom:.9rem">{txt['ins_sim']}</div>
                <div class="bar-w">
                    <div class="bar-hd"><span class="bar-nm">{txt['dataset_sim']}</span><span class="bar-nu">{avg_ds}</span></div>
                    <div class="bar-tr"><div class="bar-fi" style="width:{ds_pct}%"></div></div>
                </div>
                <div class="bar-w">
                    <div class="bar-hd"><span class="bar-nm">{txt['youm7_sim']}</span><span class="bar-nu">{avg_y7}</span></div>
                    <div class="bar-tr"><div class="bar-fi" style="width:{y7_pct}%"></div></div>
                </div>
            </div>
            <div class="icard">
                <div class="icard-l" style="margin-bottom:.9rem">{txt['ins_sources']}</div>
                {src_bars}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div style="font-weight:700;color:#fff;margin-bottom:.75rem">{txt["ins_hist"]}</div>', unsafe_allow_html=True)
        rows_html = ""
        for x in reversed(history):
            lbl = x["label"]
            if any(k in lbl for k in ["🟢", "REAL", "حقيقي", "Likely"]): bc = "hist-real"
            elif any(k in lbl for k in ["🔴", "FAKE", "مزيف"]):           bc = "hist-fake"
            else:                                                            bc = "hist-unct"
            rows_html += f"""
            <div class="hist-row">
                <span class="hist-badge {bc}">{lbl[:30]}</span>
                <span class="hist-text">{x['text']}</span>
                <span class="hist-src">{x['source']}</span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty">{txt["ins_empty"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="footer">
        <div class="f-logo">🛡️ {txt['brand']}</div>
        <div class="f-copy">{txt['footer']}</div>
    </div>""", unsafe_allow_html=True)