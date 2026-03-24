import os
import sys
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import build_vectorstore
from utils.qa_chain import build_qa_chain

load_dotenv()

st.set_page_config(
    page_title="TELUSUKO — Document Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080A0F !important;
    color: #E8E6E0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(16,185,129,0.08) 0%, transparent 55%),
        #080A0F !important;
}

/* ── Hide ALL Streamlit chrome including sidebar toggle ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="collapse sidebar"],
button[aria-label="expand sidebar"],
section[data-testid="stSidebar"] ~ div button,
div[class*="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Sidebar always visible and fixed ── */
[data-testid="stSidebar"] {
    background: rgba(10,12,18,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(20px) !important;
    transform: translateX(0);
    transition: transform 0.3s ease;
    width: 22rem !important;
    min-width: 22rem !important;
    max-width: 22rem !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}

/* ── Sidebar Toggle Buttons ── */
button[kind="secondary"] {
    font-size: 28px !important;
    padding: 0.6rem 0.8rem !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.2) !important;
    border: 1px solid rgba(99,102,241,0.5) !important;
    color: #A5B4FC !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.35) !important;
    transform: scale(1.15);
}
/* ── Brand ── */
.brand-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #E8E6E0 0%, #6366F1 50%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
}
.brand-tagline {
    font-size: 0.7rem;
    font-weight: 300;
    color: rgba(232,230,224,0.55);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #A8A6A0;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInstructions"] {
    background: #0D0F16 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(99,102,241,0.06) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: rgba(232,230,224,0.5) !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #818CF8 !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.2s ease !important;
    width: auto !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(99,102,241,0.55) !important;
    color: #A5B4FC !important;
}
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFileName"] span { color: rgba(232,230,224,0.7) !important; }
[data-testid="stFileUploaderFile"] small { color: rgba(232,230,224,0.3) !important; }
[data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    border: none !important;
    color: rgba(232,230,224,0.3) !important;
    width: auto !important;
    padding: 0.2rem !important;
}
[data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: #FCA5A5 !important;
    background: rgba(239,68,68,0.1) !important;
}

/* ── Primary button (Index) ── */
.stButton > button[kind="primary"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #818CF8, #6366F1) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── New Chat button (left col) — indigo ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    color: #A5B4FC !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.22s ease !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button:hover {
    background: rgba(99,102,241,0.24) !important;
    border-color: rgba(99,102,241,0.7) !important;
    color: #C7D2FE !important;
    box-shadow: 0 0 16px rgba(99,102,241,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Clear All button (right col) — rose red ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.38) !important;
    color: #FCA5A5 !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.22s ease !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button:hover {
    background: rgba(239,68,68,0.2) !important;
    border-color: rgba(239,68,68,0.65) !important;
    color: #FECACA !important;
    box-shadow: 0 0 16px rgba(239,68,68,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Status badges ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.9rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    margin: 0.5rem 0;
    width: 100%;
}
.status-ready   { background: rgba(16,185,129,0.1);  border: 1px solid rgba(16,185,129,0.25);  color: #34D399; }
.status-waiting { background: rgba(99,102,241,0.08);  border: 1px solid rgba(99,102,241,0.2);   color: #818CF8; }
.status-warning { background: rgba(245,158,11,0.08);  border: 1px solid rgba(245,158,11,0.2);   color: #FCD34D; }

.pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; animation: pulse 2s infinite; flex-shrink: 0;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.4;transform:scale(0.8);} }

/* ── File tag ── */
.file-tag {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.75rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px; font-size: 0.7rem;
    color: rgba(232,230,224,0.82);
    margin: 0.25rem 0; font-family: 'DM Sans', monospace;
}

/* ── Divider ── */
hr, [data-testid="stDivider"] {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1rem 0 !important;
}

/* ── Session counter ── */
.session-counter {
    font-size: 0.7rem; color: rgba(232,230,224,0.45);
    text-align: center; font-family: 'Syne', sans-serif;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Main title ── */
.main-title {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    letter-spacing: -0.02em; line-height: 1.1;
    background: linear-gradient(135deg, #E8E6E0 0%, rgba(232,230,224,0.5) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.main-subtitle {
    font-size: 0.85rem; font-weight: 300; color: rgba(232,230,224,0.7);
    letter-spacing: 0.05em; margin-bottom: 2.5rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.5rem 0 !important; animation: fadeInUp 0.3s ease forwards;
}
.stChatMessage [data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important; line-height: 1.7 !important; color: #E8E6E0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 1rem 1.25rem !important; margin-left: 3rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 1rem 1.25rem !important; margin-right: 3rem !important;
}
@keyframes fadeInUp { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:translateY(0);} }

/* ── Chat input ── */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
.stChatInputContainer, .stChatInputContainer > div {
    background: #080A0F !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background: #0D0F16 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important; transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"]:focus-within > div {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
    background: #0D0F16 !important;
}
[data-testid="stChatInput"] textarea {
    color: #E8E6E0 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; background: transparent !important;
    caret-color: #6366F1 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(232,230,224,0.22) !important; }
[data-testid="stChatInput"] button {
    background: rgba(99,102,241,0.15) !important; border-radius: 8px !important;
    color: #818CF8 !important; width: auto !important; border: none !important;
}
[data-testid="stChatInput"] button:hover {
    background: rgba(99,102,241,0.3) !important; color: #A5B4FC !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.75rem !important; color: rgba(232,230,224,0.45) !important;
    font-family: 'Syne', sans-serif !important; letter-spacing: 0.05em !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.82rem !important; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 5rem 2rem; text-align: center; gap: 1rem;
}
.empty-icon { font-size: 3rem; opacity: 0.15; }
.empty-text {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 600;
    color: rgba(232,230,224,0.55); letter-spacing: 0.05em;
}
.empty-subtext { font-size: 0.8rem; color: rgba(232,230,224,0.38); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
# ── Sidebar Toggle Button ─────────────────────────────────────────────────────

if st.session_state.sidebar_open:
    if st.button("❌", key="close_sidebar"):
        st.session_state.sidebar_open = False
else:
    if st.button("☰", key="open_sidebar"):
        st.session_state.sidebar_open = True
            
# force UI refresh (important for cloud)
st.write("")

# ── Sidebar ────────────────────────────────────────────────────────────────────
if st.session_state.sidebar_open:
    with st.sidebar:

        st.markdown("""
        <div class="brand-header">
            <div class="brand-name">✦ Telusuko</div>
            <div class="brand-tagline">Document Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            label="Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # KEEP EVERYTHING SAME BELOW 👇
        # (DO NOT CHANGE ANYTHING INSIDE)

    # ── Detect file removal ────────────────────────────────────────────────────
        if not uploaded_files:
            if st.session_state.docs_processed:
                st.session_state.vectorstore = None
                st.session_state.docs_processed = False
                st.session_state.processed_file_names = []

        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            removed_files = set(st.session_state.processed_file_names) - set(current_file_names)

            if removed_files and st.session_state.docs_processed:
                st.markdown(f"""
                <div class="status-badge status-warning">
                    <span class="pulse-dot"></span>
                    Removed: {', '.join(removed_files)}
                </div>
                """, unsafe_allow_html=True)
                st.session_state.vectorstore = None
                st.session_state.docs_processed = False

            if st.session_state.processed_file_names:
                for fname in st.session_state.processed_file_names:
                    st.markdown(f'<div class="file-tag">📄 {fname}</div>', unsafe_allow_html=True)

            files_changed = set(current_file_names) != set(st.session_state.processed_file_names)
            btn_label = "↻  Re-Process Documents" if (files_changed and st.session_state.processed_file_names) else "⚙  Start Processing"

            if st.button(btn_label, type="primary"):
                with st.spinner("Indexing..."):
                    all_docs = []
                    for file in uploaded_files:
                        docs = load_pdf(file)
                        all_docs.extend(docs)
                    chunks = split_documents(all_docs)
                    vectorstore = build_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.docs_processed = True
                    st.session_state.processed_file_names = current_file_names
                st.success(f"✓ {len(uploaded_files)} file(s) · {len(chunks)} chunks")

        # ── Status ─────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)

        if st.session_state.docs_processed:
            st.markdown("""
            <div class="status-badge status-ready">
                <span class="pulse-dot"></span>
                Ready to answer
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-waiting">
                <span class="pulse-dot"></span>
                Awaiting documents
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Voice input ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Voice Input</div>', unsafe_allow_html=True)
        components.html("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700&display=swap');
            * { box-sizing:border-box; margin:0; padding:0; } body { background:transparent; }
            .vbtn {
                display:flex; align-items:center; justify-content:center; gap:0.5rem;
                padding:0.6rem 1.4rem; background:rgba(16,185,129,0.08);
                border:1px solid rgba(16,185,129,0.3); border-radius:24px; color:#34D399;
                font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:600;
                letter-spacing:0.12em; text-transform:uppercase; cursor:pointer;
                transition:all 0.25s ease; width:100%;
            }
            .vbtn:hover { background:rgba(16,185,129,0.15); border-color:rgba(16,185,129,0.5); box-shadow:0 0 20px rgba(16,185,129,0.15); }
            .vbtn.rec { background:rgba(239,68,68,0.12); border-color:rgba(239,68,68,0.45); color:#FCA5A5; animation:rp 1.2s infinite; }
            @keyframes rp { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.3);} 50%{box-shadow:0 0 0 8px rgba(239,68,68,0);} }
            #vs { margin-top:0.45rem; font-family:'Syne',sans-serif; font-size:0.65rem; color:rgba(232,230,224,0.3); text-align:center; min-height:1rem; }
        </style>
        <button class="vbtn" id="vbtn" onclick="toggleVoice()">🎙 &nbsp;Ask with voice</button>
        <div id="vs"></div>
        <script>
            let rec=null,isRec=false;
            function init(){
                const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
                if(!SR){document.getElementById('vs').innerText='⚠ Use Chrome';return false;}
                rec=new SR();rec.continuous=false;rec.interimResults=false;rec.lang='en-US';
                rec.onstart=()=>{isRec=true;document.getElementById('vbtn').classList.add('rec');document.getElementById('vbtn').innerHTML='⏺ &nbsp;Listening...';document.getElementById('vs').innerText='Speak now...';};
                rec.onresult=(e)=>{
                    const t=e.results[0][0].transcript;
                    document.getElementById('vs').innerText='✓ '+t;
                    try{
                        const inp=window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
                        if(inp){
                            Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set.call(inp,t);
                            inp.dispatchEvent(new Event('input',{bubbles:true}));
                            setTimeout(()=>{inp.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true}));},500);
                        }
                    }catch(err){document.getElementById('vs').innerText='⚠ '+err.message;}
                };
                rec.onerror=(e)=>{document.getElementById('vs').innerText='✗ '+e.error;reset();};
                rec.onend=()=>{isRec=false;reset();};
                return true;
            }
            function reset(){document.getElementById('vbtn').classList.remove('rec');document.getElementById('vbtn').innerHTML='🎙 &nbsp;Ask with voice';}
            function toggleVoice(){
                if(!rec&&!init())return;
                if(isRec){rec.stop();}else{try{rec.start();}catch(e){init();rec.start();}}
            }
        </script>
        """, height=80)

        st.divider()

        # ── Actions ────────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✦ New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.vectorstore = None
                st.session_state.docs_processed = False
                st.session_state.processed_file_names = []
                st.rerun()

        # Stats
        if st.session_state.chat_history:
            st.divider()
            questions = len(st.session_state.chat_history) // 2
            st.markdown(
                f'<div class="session-counter">{questions} question{"s" if questions!=1 else ""} this session</div>',
                unsafe_allow_html=True
            )

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 1rem 1rem;">
    <div class="main-title">TELUSUKO DOC BOT</div>
    <div class="main-subtitle">Em Kavaloo · Upload Chesi wait chey · Adukko</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">✦</div>
        <div class="empty-text">No conversation yet</div>
        <div class="empty-subtext">Upload a PDF in the sidebar and start asking questions</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask anything about your documents...")

if user_question:
    if not st.session_state.docs_processed or st.session_state.vectorstore is None:
        st.warning("⚠ Upload and index your PDFs first using the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                qa_chain = build_qa_chain(st.session_state.vectorstore)
                result = qa_chain({"query": user_question})
                answer = result["result"]
                source_docs = result["source_documents"]

            st.markdown(answer)

            if source_docs:
                with st.expander(f"↗ {len(source_docs)} source chunks used"):
                    for i, doc in enumerate(source_docs, 1):
                        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page = doc.metadata.get("page", "?")
                        st.markdown(
                            f'<span style="font-family:Syne,sans-serif;font-size:0.7rem;'
                            f'color:rgba(232,230,224,0.4);letter-spacing:0.08em;text-transform:uppercase;">'
                            f'Chunk {i} · {filename} · Page {page}</span>',
                            unsafe_allow_html=True
                        )
                        preview = doc.page_content[:300]
                        if len(doc.page_content) > 300:
                            preview += "..."
                        st.markdown(
                            f'<div style="font-size:0.78rem;color:rgba(232,230,224,0.5);'
                            f'line-height:1.6;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05)">'
                            f'{preview}</div>',
                            unsafe_allow_html=True
                        )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})