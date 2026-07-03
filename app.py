import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from ai_research_assistant import RAGPipeline


def render_mermaid(code: str, height: int = 560) -> None:
    html = f"""
    <div class="mermaid">{code}</div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: true, theme: "default", securityLevel: "loose" }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


st.set_page_config(page_title="AI Research Assistant", page_icon="AI", layout="wide")

st.title("AI Research Assistant")

with st.sidebar:
    st.header("Document")
    api_key = st.text_input("Google API key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    uploaded_file = st.file_uploader("Upload a research paper PDF", type="pdf")

    if st.button("Process paper", type="primary", use_container_width=True):
        if not api_key:
            st.error("Add a Google API key.")
        elif uploaded_file is None:
            st.error("Upload a PDF first.")
        else:
            with st.spinner("Extracting text, chunking pages, and building ChromaDB index..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    rag = RAGPipeline(api_key=api_key)
                    chunk_count = rag.ingest_pdf(tmp_path)
                    st.session_state.rag = rag
                    st.session_state.messages = []
                    st.success(f"Indexed {chunk_count} source-backed chunks.")
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    os.unlink(tmp_path)

    if "rag" in st.session_state:
        st.success("Paper ready")


qa_tab, diagram_tab, summary_tab = st.tabs(["Q&A", "Diagrams", "Summary"])

with qa_tab:
    if "rag" not in st.session_state:
        st.info("Upload and process a PDF to start.")
    else:
        st.subheader("Ask grounded questions")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about methodology, experiments, results, or limitations"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving source chunks and drafting answer..."):
                    answer = st.session_state.rag.answer_query(prompt)
                    st.markdown(answer.text)
                    with st.expander("Retrieved citations"):
                        for citation in answer.citations:
                            st.caption(f"Page {citation.page}, chunk {citation.chunk}: {citation.preview}")
                        st.caption(f"Latency: {answer.latency_ms:.0f} ms")
                    st.session_state.messages.append({"role": "assistant", "content": answer.text})

with diagram_tab:
    if "rag" not in st.session_state:
        st.info("Upload and process a PDF to generate Mermaid diagrams.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            diagram_type = st.segmented_control("Type", ["Mind map", "Flowchart"], default="Flowchart")
        with col2:
            topic = st.text_input("Optional focus topic", placeholder="e.g. retrieval pipeline, experiments")

        if st.button("Generate diagram", type="primary"):
            with st.spinner("Generating Mermaid diagram..."):
                if diagram_type == "Mind map":
                    st.session_state.diagram_code = st.session_state.rag.generate_mindmap(topic or None)
                else:
                    st.session_state.diagram_code = st.session_state.rag.generate_flowchart(topic or None)

        if "diagram_code" in st.session_state:
            visual_tab, code_tab = st.tabs(["Visual", "Mermaid"])
            with visual_tab:
                render_mermaid(st.session_state.diagram_code)
            with code_tab:
                st.code(st.session_state.diagram_code, language="mermaid")

with summary_tab:
    if "rag" not in st.session_state:
        st.info("Upload and process a PDF to summarize it.")
    else:
        if st.button("Generate summary", type="primary"):
            with st.spinner("Summarizing paper..."):
                st.session_state.summary = st.session_state.rag.generate_summary()
        if "summary" in st.session_state:
            st.markdown(st.session_state.summary)
