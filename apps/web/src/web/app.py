"""Sistema web do projeto (ponto de entrada). Rodar com: just web

Todas as funcionalidades ficam aqui (plano experimental, seção 7.1); o front só conversa
com a API por HTTP.
"""

from pathlib import Path

import streamlit as st

from web.common import get_client

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="Triagem de melanoma — PDI", page_icon="🔬", layout="wide")

with st.sidebar:
    status = get_client().health()
    if status is None:
        st.error("API indisponível")
    else:
        st.success(f"API online · modelo: {status['model']}")

nav = st.navigation(
    {
        "Protótipo": [
            st.Page(VIEWS / "triagem.py", title="Triagem", icon=":material/stethoscope:"),
        ],
        "Experimentos": [
            st.Page(VIEWS / "simulador.py", title="Simulador", icon=":material/compare:"),
            st.Page(VIEWS / "experimentos.py", title="Experimentos", icon=":material/science:"),
            st.Page(VIEWS / "resultados.py", title="Resultados", icon=":material/monitoring:"),
            st.Page(VIEWS / "anotacao.py", title="Anotação", icon=":material/draw:"),
        ],
    }
)
nav.run()
