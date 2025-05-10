import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configurações da Página ---
st.set_page_config(
    page_title="Previsor de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# --- Título e Informações ---
st.title('🎓 Previsor de Evasão Escolar')
st.info('Este aplicativo utiliza Machine Learning para prever a probabilidade de evasão escolar de um aluno.')

# --- Carregamento dos Dados ---
@st.cache_data # Cacheia o carregamento dos dados para melhor performance
def load_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        st.warning("Por favor, verifique a URL do dataset ou a conexão com a internet.")
        return None

data_url = 'https://raw.githubusercontent.com/prof-danny-idp/machinelearning/refs/heads/main/evasao_escolar.csv'
df_raw = load_data(data_url)

if df_raw is not None:
    st.success(f"Dataset carregado com sucesso! ({df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas)")

    # --- Exploração dos Dados ---
    with st.expander("🔍 Explorar os Dados Carregados"):
        st.write('**Dados Brutos**')
        st.dataframe(df_raw.head(), use_container_width=True)

        # Assume que a última coluna é o target, ou especifique o nome
        target_column = 'Evasao_Confirmada' # Ou o nome correto da coluna alvo no seu CSV

        if target_column not in df_raw.columns:
            st.error(f"A coluna alvo '{target_column}' não foi encontrada no dataset. Verifique o nome da coluna.")
            st.stop()

        st.write('**Variáveis Preditoras (X)**')
        X_display = df_raw.drop(target_column, axis=1)
        st.dataframe(X_display.head(), use_container_width=True)

        st.write(f'**Variável Alvo (y): {target_column}**')
        y_display = df_raw[target_column]
        st.dataframe(y_display.head(), use_container_width=True)
        st.write(f"Distribuição da variável alvo: \n{y_display.value_counts(normalize=True) * 100} %")


    # --- Visualização dos Dados ---
    with st.expander("📊 Visualização dos Dados"):
        st.write("Selecione as variáveis para o gráfico de dispersão:")
        
        # Tenta encontrar colunas numéricas para os eixos do gráfico
        numeric_cols = X_display.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1_vis, col2_vis = st.columns(2)
            with col1_vis:
                x_axis = st.selectbox('Eixo X', numeric_cols, index=0 if numeric_cols else None)
            with col2_vis:
                # Garante que y_axis seja diferente de x_axis, se possível
                available_y_options = [col for col in numeric_cols if col != x_axis]
                y_axis_index = 0
                if len(available_y_options) > 1:
                     y_axis_index = 1 if len(available_y_options) >1 else 0

                y_axis = st.selectbox('Eixo Y', available_y_options, index=y_axis_index if available_y_options else None)

            if x_axis and y_axis:
                st.scatter_chart(data=df_raw, x=x_axis, y=y_axis, color=target_column, use_container_width=True)
            else:
                st.warning("Não há colunas numéricas suficientes para gerar o gráfico de dispersão.")
        else:
            st.info("Dataset não possui colunas numéricas suficientes para um gráfico de dispersão padrão.")


    # --- Input das Features na Sidebar ---
    with st.sidebar:
        st.header('📝 Insira os Dados do Aluno:')

        # Valores padrão e limites baseados na discussão anterior
        # É importante que estes nomes de colunas correspondam aos do CSV
        # Se o CSV tiver nomes diferentes, ajuste aqui e no 'data' dict abaixo.

        reprovacoes = st.slider('Reprovações Anteriores', 0, 5, 0, help="Número de vezes que o aluno foi reprovado anteriormente.")
        faltas = st.slider('Percentual de Faltas (%)', 0.0, 100.0, 10.0, step=0.5, help="Percentual de faltas no último período letivo.")
        media_notas = st.slider('Média de Notas Anterior', 0.0, 10.0, 7.0, step=0.1, help="Média das notas do aluno no ano/período anterior.")
        renda_sm = st.slider('Renda Familiar Per Capita (SM)', 0.1, 10.0, 1.0, step=0.1, help="Renda familiar dividida pelo número de pessoas, em Salários Mínimos.")
        escol_resp = st.slider('Escolaridade do Responsável (Anos)', 0, 20, 11, help="Anos de estudo completos do principal responsável pelo aluno.")
        
        aluno_trabalha_sb = st.selectbox('Aluno Trabalha?', ('Não', 'Sim'), index=0, help="O aluno exerce alguma atividade remunerada?")
        
        distorcao = st.slider('Distorção Idade-Série (Anos)', 0, 10, 0, help="Diferença entre a idade do aluno e a idade esperada para sua série.")
        
        zona_urbana_sb = st.selectbox('Reside em Zona Urbana?', ('Rural', 'Urbana'), index=1, help="O aluno reside em área urbana ou rural?")

        # Mapeamento para consistência com dados numéricos (se aplicável no CSV)
        # Se o CSV já usa 'Sim'/'Não', 'Urbana'/'Rural', este mapeamento não é estritamente necessário
        # antes do get_dummies, mas é bom para clareza se quisermos forçar 0/1 antes.
        # No entanto, para alinhar com o exemplo dos pinguins, vamos manter como string
        # e deixar o get_dummies tratar.

        data_input = {
            'Reprovacoes_Anteriores': reprovacoes,
            'Percentual_Faltas_Ultimo_Periodo': faltas,
            'Media_Notas_Ano_Anterior': media_notas,
            'Renda_Familiar_Per_Capita_SM': renda_sm,
            'Escolaridade_Responsavel_Anos': escol_resp,
            'Aluno_Trabalha': aluno_trabalha_sb, # Mantido como string
            'Distorcao_Idade_Serie': distorcao,
            'Zona_Residencia_Urbana': zona_urbana_sb # Mantido como string
        }
        
        # Verificar se as colunas do input existem no X_display (features do CSV)
        missing_cols_input = [col for col in data_input.keys() if col not in X_display.columns]
        if missing_cols_input:
            st.error(f"Erro de Mapeamento de Colunas: As seguintes colunas do input não existem no dataset carregado: {', '.join(missing_cols_input)}. Verifique os nomes das colunas no CSV e no código.")
            st.stop()

        input_df = pd.DataFrame(data_input, index=[0])

    # --- Exibição das Features de Entrada ---
    with st.expander("📥 Features de Entrada do Aluno"):
        st.write('**Dados do Aluno Inseridos:**')
        st.dataframe(input_df, use_container_width=True)

    # --- Preparação dos Dados para o Modelo ---
    # Combinar input do usuário com o dataset original para garantir consistência no encoding
    # Assegura que todas as colunas de X_display (exceto target) estejam presentes
    # e na ordem correta para input_df antes de concatenar.
    
    # Alinhar colunas do input_df com X_display
    input_df_aligned = input_df.reindex(columns=X_display.columns)
    
    combined_data_for_encoding = pd.concat([input_df_aligned, X_display], axis=0)

    # Identificar colunas categóricas (object ou category dtype) para one-hot encoding
    # Isso é mais robusto do que fixar uma lista de colunas para encode.
    categorical_cols_to_encode = combined_data_for_encoding.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Realizar o one-hot encoding
    if categorical_cols_to_encode:
        df_encoded = pd.get_dummies(combined_data_for_encoding, columns=categorical_cols_to_encode, prefix=categorical_cols_to_encode)
    else:
        df_encoded = combined_data_for_encoding.copy() # Nenhuma coluna categórica para encodar

    # Separar a linha de input (já encodada) do restante do dataset (encodado)
    input_row_processed = df_encoded.iloc[[0]]
    X_processed = df_encoded.iloc[1:]
    y_processed = df_raw[target_column] # y não precisa de encoding se já for 0/1

    with st.expander("⚙️ Dados Preparados para o Modelo"):
        st.write('**Input do Aluno (Após Encoding):**')
        st.dataframe(input_row_processed, use_container_width=True)
        st.write('**Variáveis Preditoras X (Após Encoding - Amostra):**')
        st.dataframe(X_processed.head(), use_container_width=True)
        st.write(f'**Variável Alvo y ({target_column} - Amostra):**')
        st.dataframe(y_processed.head(), use_container_width=True)
        st.write(f"Dimensões de X_processed: {X_processed.shape}")
        st.write(f"Dimensões de y_processed: {y_processed.shape}")
        st.write(f"Dimensões de input_row_processed: {input_row_processed.shape}")
        
        # Garantir que as colunas do input_row_processed correspondem a X_processed
        if not X_processed.columns.equals(input_row_processed.columns):
            st.error("Inconsistência de colunas entre os dados de treino e o input do usuário após o encoding.")
            st.write("Colunas de X_processed:", X_processed.columns.tolist())
            st.write("Colunas de input_row_processed:", input_row_processed.columns.tolist())
            
            # Tentativa de reconciliação (adicionar colunas faltantes com 0, remover extras)
            # Isso pode ocorrer se uma categoria no input não existia no X_raw original.
            # Ou se uma categoria em X_raw não estava no input.
            
            # Adicionar colunas faltantes no input_row_processed (que estão em X_processed)
            for col in X_processed.columns:
                if col not in input_row_processed.columns:
                    input_row_processed[col] = 0
            
            # Remover colunas extras do input_row_processed (que não estão em X_processed)
            for col in input_row_processed.columns:
                if col not in X_processed.columns:
                    input_row_processed = input_row_processed.drop(columns=[col])
            
            # Reordenar colunas do input_row_processed para bater com X_processed
            input_row_processed = input_row_processed[X_processed.columns]

            st.warning("Tentativa de reconciliação de colunas aplicada. Verifique os dados abaixo.")
            st.write('**Input do Aluno (Após Reconciliação e Encoding):**')
            st.dataframe(input_row_processed, use_container_width=True)
            
            if not X_processed.columns.equals(input_row_processed.columns):
                 st.error("Falha na reconciliação de colunas. O modelo pode não funcionar corretamente.")
                 st.stop()


    # --- Treinamento do Modelo e Inferência ---
    st.subheader('🚀 Resultado da Previsão')
    try:
        # Dividir os dados carregados em treino e teste para avaliação (opcional, mas bom para ter uma ideia da performance)
        # X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed if y_processed.nunique() > 1 else None)

        # Treinar o modelo com TODOS os dados carregados (X_processed, y_processed) para a predição no input do usuário
        model = RandomForestClassifier(random_state=42, class_weight='balanced') # class_weight para datasets desbalanceados
        model.fit(X_processed, y_processed)

        # Fazer predição na linha de input do usuário
        prediction = model.predict(input_row_processed)
        prediction_proba = model.predict_proba(input_row_processed)

        # --- Exibir Resultados da Predição ---
        status_aluno = np.array(['Não Evadirá', 'Evadirá'])
        
        df_prediction_proba = pd.DataFrame(prediction_proba, columns=status_aluno)

        st.write('**Probabilidades de Evasão:**')
        st.dataframe(df_prediction_proba,
                     column_config={
                         'Não Evadirá': st.column_config.ProgressColumn(
                             'Não Evadirá',
                             help="Probabilidade do aluno NÃO evadir.",
                             format="%.2f%%", # Formato de porcentagem
                             min_value=0,
                             max_value=1, # Probabilidade é entre 0 e 1
                         ),
                         'Evadirá': st.column_config.ProgressColumn(
                             'Evadirá',
                             help="Probabilidade do aluno EVADIR.",
                             format="%.2f%%",
                             min_value=0,
                             max_value=1,
                         ),
                     }, hide_index=True, use_container_width=True)

        # Destacar a predição
        resultado_final = status_aluno[prediction][0]
        if resultado_final == 'Evadirá':
            st.error(f'**Predição Final: {resultado_final}** 😟')
            st.warning("Atenção: Este aluno apresenta risco de evasão. Considere ações preventivas.")
        else:
            st.success(f'**Predição Final: {resultado_final}** 😊')
            st.balloons()

        # --- (Opcional) Feature Importance ---
        with st.expander("💡 Importância das Features (do modelo treinado)"):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_processed.columns
                df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                df_importance = df_importance.sort_values(by='Importance', ascending=False)
                
                st.bar_chart(df_importance.set_index('Feature'), use_container_width=True)
            else:
                st.info("O modelo selecionado não suporta a extração direta de importância das features.")

    except Exception as e:
        st.error(f"Ocorreu um erro durante o treinamento ou predição: {e}")
        st.error("Verifique se os dados de entrada estão corretos e se o dataset carregado é adequado.")
        # st.exception(e) # Para debugging mais detalhado

else:
    st.warning("O dataset não pôde ser carregado. O aplicativo não pode continuar.")

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido como exemplo de app de Machine Learning com Streamlit.")

