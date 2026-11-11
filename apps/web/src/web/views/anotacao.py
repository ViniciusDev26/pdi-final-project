"""Página Anotação: máscaras de referência das listas de anotação (plano, seção 5.8).

O anotador clica nos pontos do contorno da lesão sobre a imagem ampliada; o polígono
fechado vira a máscara, rasterizada pela API. A segmentação automática não é mostrada,
para não induzir a anotação (a referência precisa ser independente do algoritmo).
"""

import io

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from web.common import api_error_message, b64_image, get_client

SCALE = 2  # imagem 300 × 300 exibida em 600 × 600 para clicar com precisão
SET_LABELS = {"val": "Validação (ajuste, 20 imagens)", "teste": "Teste (avaliação, 60 imagens)"}

client = get_client()

st.title("Anotação")
st.caption(
    "Desenhe o contorno de toda a área pigmentada da lesão, inclusive a periferia mais "
    "clara. Pelos sobre a lesão fazem parte dela; vinheta, régua e marcações ficam de fora. "
    "Clique nos pontos do contorno em sequência: o polígono é fechado automaticamente."
)

c1, c2 = st.columns(2)
conjunto = c1.selectbox("Conjunto", list(SET_LABELS), format_func=SET_LABELS.get, key="ann_set")

try:
    listing = client.annotations(conjunto)
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

annotator = c2.selectbox("Anotador", listing["annotators"], key="ann_annotator")

with st.expander("Progresso e concordância entre anotadores"):
    prog = pd.DataFrame(listing["progress"])
    st.dataframe(prog, hide_index=True, width="stretch")
    if listing["inter_annotator"]:
        st.metric("Dice médio entre anotadores", f"{listing['inter_annotator_mean']:.3f}")
        st.dataframe(pd.DataFrame(listing["inter_annotator"]), hide_index=True)
    else:
        st.caption("Dice entre anotadores: nenhuma imagem compartilhada anotada por dois ainda.")

mine = [i for i in listing["items"] if annotator in i["annotators"]]
if not mine:
    st.info("Nenhuma imagem atribuída a este anotador neste conjunto.")
    st.stop()

done = sum(i["done"][annotator] for i in mine)
st.progress(done / len(mine), text=f"{annotator}: {done} de {len(mine)} anotadas")


def _item_label(item: dict) -> str:
    mark = "✅" if item["done"][annotator] else "⬜"
    shared = " · compartilhada" if item["compartilhada"] else ""
    return f"{mark} {item['name']}{shared}"


pending = [i["name"] for i in mine if not i["done"][annotator]]
default = [i["name"] for i in mine].index(pending[0]) if pending else 0
name = st.selectbox(
    "Imagem",
    [i["name"] for i in mine],
    index=default,
    key=f"ann_img_{conjunto}_{annotator}",
    format_func=lambda n: _item_label(next(i for i in mine if i["name"] == n)),
)

points_key = f"ann_pts_{conjunto}_{annotator}_{name}"
click_key = f"ann_click_{conjunto}_{annotator}_{name}"
points: list[list[float]] = st.session_state.setdefault(points_key, [])

try:
    data = client.annotation_image(conjunto, name, annotator)
except httpx.HTTPError as exc:
    st.error(api_error_message(exc))
    st.stop()

base = Image.open(io.BytesIO(b64_image(data["image"]))).convert("RGB")


def _render(points: list[list[float]], saved_mask: bytes | None) -> Image.Image:
    img = base.resize((base.width * SCALE, base.height * SCALE), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")
    if saved_mask is not None and not points:
        mask = np.asarray(Image.open(io.BytesIO(saved_mask)).convert("L")) > 127
        overlay = Image.fromarray((mask * 90).astype(np.uint8)).resize(img.size)
        img.paste(Image.new("RGB", img.size, (0, 120, 255)), mask=overlay)
    scaled = [(x * SCALE, y * SCALE) for x, y in points]
    if len(scaled) >= 3:
        draw.polygon(scaled, fill=(255, 220, 0, 50))
    if len(scaled) >= 2:
        draw.line([*scaled, scaled[0]] if len(scaled) >= 3 else scaled, fill="yellow", width=2)
    for x, y in scaled:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red", outline="white")
    return img


saved = b64_image(data["mask"]) if data["mask"] else None
col_img, col_side = st.columns([3, 1])
with col_img:
    click = streamlit_image_coordinates(_render(points, saved), key=f"ann_canvas_{points_key}")
if click and click.get("unix_time") != st.session_state.get(click_key):
    st.session_state[click_key] = click.get("unix_time")
    factor = base.width / click["width"]  # coordenadas exibidas → imagem 300 × 300
    points.append([round(click["x"] * factor, 1), round(click["y"] * factor, 1)])
    st.rerun()

with col_side:
    item = next(i for i in mine if i["name"] == name)
    st.markdown(f"**{name}**")
    st.caption(
        f"Rótulo: {'maligna' if item['label'] == 'malignant' else 'benigna'} · "
        f"{'com' if item['vinheta'] else 'sem'} vinheta"
    )
    if saved and not points:
        st.success("Máscara salva (azul). Clique na imagem para redesenhar.")
    st.write(f"Pontos do contorno: {len(points)}")
    if st.button("Desfazer último ponto", disabled=not points, width="stretch"):
        points.pop()
        st.rerun()
    if st.button("Limpar", disabled=not points, width="stretch"):
        points.clear()
        st.rerun()
    if st.button(
        "Salvar máscara",
        type="primary",
        disabled=len(points) < 3,
        width="stretch",
        icon=":material/save:",
    ):
        try:
            client.save_annotation(conjunto, name, annotator, points)
        except httpx.HTTPError as exc:
            st.error(api_error_message(exc))
        else:
            points.clear()
            st.toast(f"Máscara de {name} salva")
            st.rerun()
