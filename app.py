
"""Streamlit chat agent on top of the ChromaDB knowledge base.

Pipeline (same as in 02_rag_engine.ipynb):
rewrite_query -> embedding_retrieval -> rerank_contexts -> generate_answer

Run with:  streamlit run app.py
"""
import os
import re
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

from config.settings import DATA_DIR, RESULTS_TAB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = 'openai/text-embedding-3-small'
CHAT_MODEL = 'anthropic/claude-sonnet-5.5'
RETRIEVAL_K = 20       # chunks retrieved per query
RERANKED_K = 6         # chunks kept after reranking
HISTORY_MESSAGES = 6   # last 6 messages = 3 question/answer pairs sent to the LLM
AVATARS = {'user': '🙋', 'assistant': '⛏️'}

data_dir = Path(DATA_DIR)
results_tab = Path(RESULTS_TAB)

st.set_page_config(page_title='Project Miner Assistant', page_icon='💬', layout='centered')


# ---------------------------------------------------------------------------
# Cached resources (created once per server session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    load_dotenv(override=True)
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        st.error('Missing OPENROUTER_API_KEY')
        st.stop()
    return OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1')


@st.cache_resource
def get_collection():
    chroma = PersistentClient(path=str(data_dir / 'chroma_db'))
    return chroma.get_collection(name='knowledge_base')


client = get_client()
collection = get_collection()

# ---------------------------------------------------------------------------
# Conversation memory: [{'timestamp', 'question', 'answer'}]
# Stored in session_state so it survives Streamlit reruns
# ---------------------------------------------------------------------------
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'debug' not in st.session_state:
    st.session_state.debug = {}   # index of turn -> queries and contexts (for the expander)

conversation = st.session_state.conversation

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
REWRITE_PROMPT = '''
Rewrite the user question into short English search queries for a knowledge base
of project READMEs, notebooks, code and figures.

Goals:
- Preserve the original intent
- The knowledge base is in English, so write the queries in English regardless of the question language
- Use the conversation history only to resolve references like "it", "that", "the previous one", so every query is fully self-contained
- If the question is unrelated to the history, ignore the history
- Expand vague wording into retrieval-friendly terminology
- Split multi-part questions into separate queries, one per line
- Keep project names, numbers and technical terms exactly (e.g. lift, anomaly, DAX)
- Add likely technical synonyms (e.g. "computed" -> feature, formula, score, method) only when useful
- If user requests scoring, keep the requested scale
- If the user specifies a persona or expertise level, preserve it
- If the user asks to evaluate a project, include likely evidence dimensions:
  project goal, business value, data, methodology, experiments, reproducibility, strengths, weaknesses, limitations, and overall verdict

Return only the queries, one per line.
'''

RERANK_PROMPT = '''You are a relevance reranker.

Select the {top_k} candidates most useful for answering the question, best first.
- Cover EVERY part of the question (e.g. both "how was X computed" and "what is Y")
- Prefer concrete definitions, formulas, methods and results over charts and overviews
- Use the 1-based numbers shown in brackets

Return only a JSON list of numbers, e.g. [3, 1, 7]'''

SYSTEM_PROMPT = '''
You answer questions about the user's projects.

Rules:
- ALWAYS answer in the same language as the user's original question
- Answer strictly from the provided context. Never invent metrics, causes or conclusions
- If the context does not contain the answer, say so in one short sentence
- If only part of the question is covered, answer that part and state what is not stated
- Use project and source information only to understand the context
- Never treat the same PROJECT name as more than one project
- Use the earlier conversation only to understand follow-up questions; the facts must come from the provided context
- Do NOT mention chunk numbers, project names, file names or sources in the answer. Write only the content (text, numbers)
- Do NOT ask follow-up questions or add closing sentences

Output:
- Start with a short plain-language summary for a non-expert
- If the user explicitly requests an expert, manager, scoring, or detailed review, give a professional interpretation with evidence from the context
- Keep the answer practical, specific, and grounded in evidence
- Use short bullets when useful
'''


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def rewrite_query(question, conversation=None):
    # Short history block: last turns only, answers truncated to 500 characters
    history_text = ''
    if conversation:
        turns = conversation[-(HISTORY_MESSAGES // 2):]
        history_text = 'Conversation history:\n' + '\n'.join(
            f"user: {t['question']}\nassistant: {t['answer'][:500]}" for t in turns
        ) + '\n\n'

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': REWRITE_PROMPT},
                {'role': 'user', 'content': f'{history_text}Question:\n{question}'}
            ],
            temperature=0.1,   # near-deterministic rewriting
            max_tokens=500
        )
        # One query per line; strip bullet characters the model may add
        lines = (resp.choices[0].message.content or '').splitlines()
        queries = [q.strip('-• ').strip() for q in lines if q.strip()]
        return queries or [question]
    except Exception as e:
        print(f'rewrite_query failed: {e}')
        return [question]


def embedding_retrieval(queries, k=RETRIEVAL_K):
    # One API call for all queries
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=queries)
    embeddings = [d.embedding for d in emb.data]

    # One ChromaDB request for all vectors
    res = collection.query(
        query_embeddings=embeddings,
        n_results=k,
        include=['documents', 'metadatas', 'distances'],
    )

    # Separate list of chunks for each query (best first); score = 1 - distance
    per_query = [
        [{
            'text': doc,
            'project': meta.get('project', ''),
            'source': meta.get('source', ''),
            'headline': meta.get('headline', ''),
            'score': 1 - dist
        }
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        for docs, metas, dists in zip(
            res['documents'], res['metadatas'], res['distances']
        )
    ]

    # Merge in rank order (1st of every query, then 2nd, ...) and skip duplicate texts
    seen, results = set(), []
    for rank in range(k):
        for chunks in per_query:
            if rank < len(chunks) and chunks[rank]['text'] not in seen:
                seen.add(chunks[rank]['text'])
                results.append(chunks[rank])
                if len(results) == k:
                    return results
    return results


def rerank_contexts(question, candidates, top_k=RERANKED_K):
    if not candidates:
        return []

    # Number the candidates so the LLM can answer with indices only
    packed = '\n\n'.join(
        f"[{i}] SOURCE: {c['source']}\nHEADLINE: {c['headline']}\nTEXT: {c['text'][:600]}"
        for i, c in enumerate(candidates, 1)
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': RERANK_PROMPT.format(top_k=top_k)},
                {'role': 'user', 'content': f'Question:\n{question}\n\nCandidates:\n{packed}'},
            ],
            temperature=0.0,   # deterministic ranking
            max_tokens=1000
        )
        # Extract the JSON-like list, e.g. [3, 1, 7]
        raw = resp.choices[0].message.content or ''
        m = re.search(r'\[[\d,\s]*\]', raw)
        indices = [int(n) for n in re.findall(r'\d+', m.group(0))] if m else []
    except Exception as e:
        print(f'rerank_contexts failed: {e}')
        return candidates[:top_k]

    # Validate indices, remove duplicates and keep the LLM order
    seen, result = set(), []
    for i in indices:
        if 1 <= i <= len(candidates) and i not in seen:
            seen.add(i)
            result.append(candidates[i - 1])
    return result[:top_k] or candidates[:top_k]


def generate_answer(question, contexts, conversation=None):
    if not contexts:
        return 'No relevant context was found.'

    # Numbered context block built from the reranked chunks
    context_block = '\n\n'.join(
        f"[{i}] SOURCE: {c['source']}\nHEADLINE: {c['headline']}\nTEXT: {c['text']}"
        for i, c in enumerate(contexts, 1)
    )

    # Messages: system prompt -> recent history (questions and answers only) -> current question with context
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for t in (conversation or [])[-(HISTORY_MESSAGES // 2):]:
        messages.append({'role': 'user', 'content': t['question']})
        messages.append({'role': 'assistant', 'content': t['answer']})
    messages.append({'role': 'user', 'content': f'Question:\n{question}\n\nContext:\n{context_block}'})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,   # low variation, factual answers with slightly natural wording
            max_tokens=3000
        )
        return (resp.choices[0].message.content or '').strip()
    except Exception as e:
        return f'generate_answer failed: {e}'


def chat(question):
    """Run the full pipeline, store the turn in memory and return (answer, debug info)."""
    queries = rewrite_query(question, conversation)
    candidates = embedding_retrieval(queries)
    contexts = rerank_contexts(' | '.join(queries), candidates)
    answer = generate_answer(question, contexts, conversation)

    # Save the turn (contexts are not stored in the history, only question and answer)
    conversation.append({
        'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M:%S'),
        'question': question,
        'answer': answer
    })
    debug = {'queries': queries, 'n_candidates': len(candidates), 'contexts': contexts}
    return answer, debug


# ---------------------------------------------------------------------------
# The two helper functions exposed as sidebar buttons
# ---------------------------------------------------------------------------
def reset_chat():
    # Start a new topic: forget the previous conversation and debug info
    conversation.clear()
    st.session_state.debug.clear()
    st.session_state.pop('export_msg', None)


def export_conversation():
    # Save the conversation to JSON and TXT in results/tables, return the file paths
    results_tab.mkdir(parents=True, exist_ok=True)
    base_name = results_tab / f'conversation_{datetime.now():%d.%m.%Y_%H-%M}'

    # JSON: full structured history
    Path(f'{base_name}.json').write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2), encoding='utf-8')

    # TXT: readable blocks (time, question, answer) separated by a dashed line
    Path(f'{base_name}.txt').write_text(
        f"\n{'-' * 40}\n\n".join(
            f"[{t['timestamp']}]\nQ: {t['question']}\n\nA: {t['answer']}"
            for t in conversation),
        encoding='utf-8'
    )
    return f'{base_name}.json', f'{base_name}.txt'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header('Controls')
    st.caption(f'Knowledge base: {collection.count()} chunks')

    show_debug = st.toggle('Show retrieval details', value=False,
                           help='Show rewritten queries and selected contexts under each answer')

    st.button('♻️ New chat', on_click=reset_chat, use_container_width=True,
              help='Forget the conversation and start a new topic')

    if st.button('💾 Export conversation', use_container_width=True,
                 disabled=not conversation,
                 help='Save the chat to JSON and TXT in the results/tables folder'):
        json_path, txt_path = export_conversation()
        st.session_state.export_msg = f'Saved:\n{json_path}\n{txt_path}'

    if st.session_state.get('export_msg'):
        st.success(st.session_state.export_msg)

    st.divider()
    st.caption(f'Conversation memory: last {HISTORY_MESSAGES // 2} question/answer pairs')

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title('💬 Project Miner Assistant')
st.caption(
    "Learn more about Tomas Sojka's data science and engineering projects."
    'Answers are drawn directly from the project documentation. Example questions:\n\n'
    '- "Which projects use time series forecasting?"\n'
    '- "What projects are in the database?"\n'
    '- "What technologies were used in the anomaly classification project?"\n'
    '- "Tell me the basic information about market-event-impact."\n'
    '- "How was the lift calculated?"\n'
)


def render_debug(info):
    # Expander with the rewritten queries and the contexts used for the answer
    with st.expander('Retrieval details'):
        st.markdown('**Queries**')
        for q in info['queries']:
            st.markdown(f'- {q}')
        st.markdown(f"**Contexts** ({len(info['contexts'])} of {info['n_candidates']} candidates)")
        for i, c in enumerate(info['contexts'], 1):
            st.markdown(f"{i}. `{c['project']}` | `{c['source']}` | {c['headline']} | score={c['score']:.4f}")


# Replay the history on every rerun
for idx, turn in enumerate(conversation):
    with st.chat_message('user', avatar=AVATARS['user']):
        st.markdown(turn['question'])
    with st.chat_message('assistant', avatar=AVATARS['assistant']):
        st.markdown(turn['answer'])
        if show_debug and idx in st.session_state.debug:
            render_debug(st.session_state.debug[idx])

# New question
if question := st.chat_input('Ask a question about the portfolio projects...'):
    with st.chat_message('user', avatar=AVATARS['user']):
        st.markdown(question)
    with st.chat_message('assistant', avatar=AVATARS['assistant']):
        with st.spinner('Searching the knowledge base...'):
            answer, debug = chat(question)
        st.markdown(answer)
        st.session_state.debug[len(conversation) - 1] = debug
        if show_debug:
            render_debug(debug)
