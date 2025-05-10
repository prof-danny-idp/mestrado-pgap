import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚨 Detecção de Fraude em Leilões")

st.info("Este app utiliza algoritmos de Machine Learning para prever a probabilidade de uma atividade fraudulenta em leilões.")

# Carregar dados
url = 'https://raw.githubusercontent.com/prof-danny-idp/machinelearning/refs/heads/main/fraude_leilao_pt.csv'
df = pd.read_csv(url)

with st.expander("Visualizar dados"):
    st.dataframe(df)

# Separar X e y
X_raw = df.drop('Classe', axis=1)
y = df['Classe']

# Sidebar - Entrada de dados do usuário
st.sidebar.header("Entrada das variáveis")

def entrada_usuario():
    ID_Registro = 0
    ID_Leilao = 0
    ID_Licitante = 0
    Tendencia_Licitante = st.sidebar.slider("Tendência do Licitante", 0.0, 1.0, 0.5)
    Proporcao_Lances = st.sidebar.slider("Proporção de Lances", 0.0, 1.0, 0.5)
    Ultrapassagem_Sucessiva = st.sidebar.slider("Ultrapassagem Sucessiva", 0.0, 1.0, 0.5)
    Ultimo_Lance = st.sidebar.slider("Último Lance", 0.0, 1.0, 0.5)
    Numero_Lances = st.sidebar.slider("Número de Lances", float(X_raw['Numero_Lances'].min()), float(X_raw['Numero_Lances'].max()), float(X_raw['Numero_Lances'].mean()))
    Preco_Inicial_Medio = st.sidebar.slider("Preço Inicial Médio", float(X_raw['Preco_Inicial_Medio'].min()), float(X_raw['Preco_Inicial_Medio'].max()), float(X_raw['Preco_Inicial_Medio'].mean()))
    Lances_Iniciais = st.sidebar.slider("Lances Iniciais", 0.0, 1.0, 0.5)
    Taxa_Vitorias = st.sidebar.slider("Taxa de Vitórias", 0.0, 1.0, 0.5)
    Duracao_Leilao = st.sidebar.slider("Duração do Leilão", float(X_raw['Duracao_Leilao'].min()), float(X_raw['Duracao_Leilao'].max()), float(X_raw['Duracao_Leilao'].mean()))
    
    dados = {
        'ID_Leilao': ID_Leilao,
        'ID_Licitante': ID_Licitante,
        'Tendencia_Licitante': Tendencia_Licitante,
        'Proporcao_Lances': Proporcao_Lances,
        'Ultrapassagem_Sucessiva': Ultrapassagem_Sucessiva,
        'Ultimo_Lance': Ultimo_Lance,
        'Numero_Lances': Numero_Lances,
        'Preco_Inicial_Medio': Preco_Inicial_Medio,
        'Lances_Iniciais': Lances_Iniciais,
        'Taxa_Vitorias': Taxa_Vitorias,
        'Duracao_Leilao': Duracao_Leilao
    }
    return pd.DataFrame(dados, index=[0])

input_df = entrada_usuario()

# Combinar entrada com os dados para garantir o mesmo formato
input_X = pd.concat([input_df, X_raw], axis=0).reset_index(drop=True)
X = input_X.drop(['ID_Leilao', 'ID_Licitante'], axis=1)
input_row = X.iloc[[0]]
X_model = X.drop(index=0)

# Filtrar apenas colunas numéricas para X
X_raw_num = X_raw.select_dtypes(include=[np.number])

# Garantir que input_df tenha as mesmas colunas numéricas
input_df_num = input_df[X_raw_num.columns]

# Concatenar entrada do usuário com o restante para manter compatibilidade
X_concat = pd.concat([input_df_num, X_raw_num], axis=0).reset_index(drop=True)

# Separar entrada do usuário e dados para treino
input_row = X_concat.iloc[[0]]
X_model = X_concat.iloc[1:]

# Treinar modelos
models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Árvore de Decisão': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}
metricas = {}

for nome, modelo in models.items():
    modelo.fit(X_model, y)
    y_pred = modelo.predict(X_model)
    proba = modelo.predict_proba(input_row)[0][1]
    results[nome] = proba
    
    # Métricas
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    metricas[nome] = {'acuracia': acc, 'precisao': prec, 'recall': rec, 'matriz': cm}

# Mostrar resultado de previsão
st.subheader("🔍 Probabilidade de Fraude Estimada")

for nome, proba in results.items():
    st.write(f"**{nome}**")
    st.progress(proba)
    st.write(f"Probabilidade de fraude: `{proba:.2%}`")

# Mostrar entrada
with st.expander("📝 Entrada do usuário"):
    st.write(input_df)

# Relatório de desempenho dos modelos
with st.expander("📊 Relatório dos Modelos"):
    for nome, met in metricas.items():
        st.markdown(f"### {nome}")
        st.write(f"Acurácia: `{met['acuracia']:.2%}`")
        st.write(f"Precisão: `{met['precisao']:.2%}`")
        st.write(f"Recall: `{met['recall']:.2%}`")
        
        # Matriz de confusão como gráfico
        fig, ax = plt.subplots()
        sns.heatmap(met['matriz'], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Fraude"], yticklabels=["Normal", "Fraude"], ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusão")
        st.pyplot(fig)
