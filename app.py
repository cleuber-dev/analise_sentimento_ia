import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# Dados para "treinamento" da IA
# =========================
data = {
    'texto': [
        'Eu adorei esse filme',
        'Esse filme é horrível',
        'Muito bom, recomendo',
        'Não gostei, muito ruim',
        'Excelente atuação',
        'Péssimo roteiro',
        'Gostei bastante do filme',
        'O filme foi muito bom',
        'Não recomendo esse filme',
        'Filme ruim e chato',
        'Atuação maravilhosa',
        'História fraca e sem graça'
    ],
    'sentimento': [
        'positivo', 'negativo', 'positivo', 'negativo',
        'positivo', 'negativo', 'positivo', 'positivo',
        'negativo', 'negativo', 'positivo', 'negativo'
    ]
}

df = pd.DataFrame(data)

# =========================
# Treinamento do modelo
# =========================
X = df['texto']
y = df['sentimento']

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_vectorizado = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorizado, y)

# =========================
# Interface do projeto
# =========================
st.title("Analisador de Sentimentos")
st.write("Digite um texto e descubra se o sentimento é positivo ou negativo.")

texto_usuario = st.text_area("Texto aqui: 👇")

if st.button("Analisar sentimento"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, digite algum texto.")
    else:
        texto_vectorizado = vectorizer.transform([texto_usuario])
        resultado = model.predict(texto_vectorizado)

        if resultado[0] == "positivo":
            st.success("😊 Sentimento POSITIVO")
        else:
            st.error("😞 Sentimento NEGATIVO")
