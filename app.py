import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentimento de Avaliações", page_icon="🛒")

@st.cache_resource
def carregar_modelo():
    return pipeline("text-classification", model="VictorHGomes/bertimbau-sentimento-olist")

clf = carregar_modelo()

st.title("🛒 Análise de Sentimento de Avaliações")
st.write(
    "BERTimbau ajustado com ~27 mil avaliações reais de e-commerce brasileiro (Olist). "
    "F1 macro de **0,937** no teste, com **29% menos erros** que um baseline TF-IDF. "
    "Experimente frases com negações e contrastes!"
)

exemplos = [
    "",
    "Chegou antes do prazo, adorei!",
    "Produto veio quebrado e ninguém responde",
    "Não é ruim, até que gostei",
    "Hidrata bem, porém não alisa muito",
    "Só há fatos positivos, não houve transtornos.",
]
exemplo = st.selectbox("Escolha um exemplo ou escreva abaixo:", exemplos)
texto = st.text_area("Avaliação", value=exemplo, height=100, placeholder="Escreva uma avaliação de compra...")

if st.button("Classificar", type="primary") and texto.strip():
    resultados = clf(texto, top_k=None)
    melhor = resultados[0]
    if melhor["label"] == "positivo":
        st.success(f"👍 **Positivo** ({melhor['score']:.0%} de confiança)")
    else:
        st.error(f"👎 **Negativo** ({melhor['score']:.0%} de confiança)")
    for r in resultados:
        st.progress(r["score"], text=f"{r['label']}: {r['score']:.1%}")

st.caption("[Código no GitHub](https://github.com/VictorHGomes/analise-sentimento-olist) · "
           "[Modelo no Hugging Face](https://huggingface.co/VictorHGomes/bertimbau-sentimento-olist)")
