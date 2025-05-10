import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configurações da Página ---
st.set_page_config(
    page_title="Previsor de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

# --- Título e Informações ---
st.title('🎓 Previsor de Evasão Escolar')
st.info('Este aplicativo utiliza Machine Learning para prever a probabilidade de evasão escolar de um aluno, comparando dois modelos: Árvore de Decisão e Random Forest.')

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
        st.write('**Dados Brutos (Amostra)**')
        st.dataframe(df_raw.head(), use_container_width=True)

        target_column = 'Evasao_Confirmada'

        if target_column not in df_raw.columns:
            st.error(f"A coluna alvo '{target_column}' não foi encontrada no dataset. Verifique o nome da coluna.")
            st.stop()

        st.write('**Variáveis Preditoras (X) (Amostra)**')
        X_display = df_raw.drop(target_column, axis=1)
        st.dataframe(X_display.head(), use_container_width=True)

        st.write(f'**Variável Alvo (y): {target_column} (Amostra)**')
        y_display = df_raw[target_column]
        st.dataframe(y_display.head().to_frame(), use_container_width=True)
        st.write(f"Distribuição da variável alvo (%): \n{y_display.value_counts(normalize=True).mul(100).round(2).astype(str) + '%'}")


    # --- Input das Features na Sidebar ---
    with st.sidebar:
        st.header('📝 Insira os Dados do Aluno:')
        reprovacoes = st.slider('Reprovações Anteriores', 0, 5, 0, help="Número de vezes que o aluno foi reprovado anteriormente.")
        faltas = st.slider('Percentual de Faltas (%)', 0.0, 100.0, 10.0, step=0.5, help="Percentual de faltas no último período letivo.")
        media_notas = st.slider('Média de Notas Anterior', 0.0, 10.0, 7.0, step=0.1, help="Média das notas do aluno no ano/período anterior.")
        renda_sm = st.slider('Renda Familiar Per Capita (SM)', 0.1, 10.0, 1.0, step=0.1, help="Renda familiar dividida pelo número de pessoas, em Salários Mínimos.")
        escol_resp = st.slider('Escolaridade do Responsável (Anos)', 0, 20, 11, help="Anos de estudo completos do principal responsável pelo aluno.")
        aluno_trabalha_sb = st.selectbox('Aluno Trabalha?', ('Não', 'Sim'), index=0, help="O aluno exerce alguma atividade remunerada?")
        distorcao = st.slider('Distorção Idade-Série (Anos)', 0, 10, 0, help="Diferença entre a idade do aluno e a idade esperada para sua série.")
        zona_urbana_sb = st.selectbox('Reside em Zona Urbana?', ('Rural', 'Urbana'), index=1, help="O aluno reside em área urbana ou rural?")

        data_input_dict = {
            'Reprovacoes_Anteriores': reprovacoes,
            'Percentual_Faltas_Ultimo_Periodo': faltas,
            'Media_Notas_Ano_Anterior': media_notas,
            'Renda_Familiar_Per_Capita_SM': renda_sm,
            'Escolaridade_Responsavel_Anos': escol_resp,
            'Aluno_Trabalha': aluno_trabalha_sb,
            'Distorcao_Idade_Serie': distorcao,
            'Zona_Residencia_Urbana': zona_urbana_sb
        }
        
        missing_cols_input = [col for col in data_input_dict.keys() if col not in X_display.columns]
        if missing_cols_input:
            st.error(f"Erro de Mapeamento de Colunas: As seguintes colunas do input não existem no dataset carregado: {', '.join(missing_cols_input)}. Verifique os nomes das colunas no CSV e no código.")
            st.stop()
        
        input_df_user = pd.DataFrame(data_input_dict, index=[0])

    # --- Exibição das Features de Entrada ---
    with st.expander("📥 Features de Entrada do Aluno (Conforme Inserido)"):
        st.write('**Dados do Aluno Inseridos:**')
        st.dataframe(input_df_user, use_container_width=True)

    # --- Preparação dos Dados para o Modelo (Encoding) ---
    # Alinhar colunas do input_df_user com X_display (features do CSV)
    input_df_aligned = input_df_user.reindex(columns=X_display.columns)
    combined_data_for_encoding = pd.concat([input_df_aligned, X_display], axis=0, ignore_index=True)

    categorical_cols_to_encode = combined_data_for_encoding.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols_to_encode:
        df_encoded_full = pd.get_dummies(combined_data_for_encoding, columns=categorical_cols_to_encode, prefix=categorical_cols_to_encode)
    else:
        df_encoded_full = combined_data_for_encoding.copy()

    input_row_processed = df_encoded_full.iloc[[0]]
    X_processed = df_encoded_full.iloc[1:]
    y_processed = df_raw[target_column]

    # Garantir que as colunas do input_row_processed correspondem a X_processed
    if not X_processed.columns.equals(input_row_processed.columns):
        st.error("Inconsistência de colunas detectada após o encoding inicial.")
        # Adicionar colunas faltantes no input_row_processed (que estão em X_processed) com valor 0
        for col in X_processed.columns:
            if col not in input_row_processed.columns:
                input_row_processed[col] = 0
        
        # Remover colunas extras do input_row_processed (que não estão em X_processed)
        cols_to_drop_from_input = [col for col in input_row_processed.columns if col not in X_processed.columns]
        if cols_to_drop_from_input:
            input_row_processed = input_row_processed.drop(columns=cols_to_drop_from_input)
        
        # Reordenar colunas do input_row_processed para bater com X_processed
        input_row_processed = input_row_processed[X_processed.columns]
        
        if X_processed.columns.equals(input_row_processed.columns):
            st.warning("Tentativa de reconciliação de colunas aplicada ao input do usuário.")
        else:
            st.error("Falha crítica na reconciliação de colunas. O modelo pode não funcionar corretamente.")
            st.write("Colunas de X_processed:", X_processed.columns.tolist())
            st.write("Colunas de input_row_processed (após tentativa de ajuste):", input_row_processed.columns.tolist())
            st.stop()


    # --- Divisão dos dados para avaliação dos modelos ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=0.3, 
            random_state=42, 
            stratify=y_processed if y_processed.nunique() > 1 else None
        )
    except Exception as e:
        st.error(f"Erro ao dividir os dados em treino e teste: {e}")
        st.info("Isso pode ocorrer se o dataset for muito pequeno ou a variável alvo tiver pouca variação.")
        st.stop()

    # --- Treinamento, Avaliação dos Modelos e Previsão para o Aluno ---
    st.header('📊 Avaliação dos Modelos e Previsão para o Aluno Inserido')

    model_configs = {
        'Árvore de Decisão': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    }
    
    status_aluno_map = {0: 'Não Evadirá', 1: 'Evadirá'} # Mapeamento para nomes das classes

    for model_name, model_eval_instance in model_configs.items():
        st.subheader(f"--- Modelo: {model_name} ---")

        # 1. Avaliação do Modelo (usando X_train, X_test)
        st.write("**1. Avaliação do Modelo (com dados de teste):**")
        try:
            model_eval_instance.fit(X_train, y_train)
            y_pred_test = model_eval_instance.predict(X_test)

            # Matriz de Confusão
            cm = confusion_matrix(y_test, y_pred_test)
            df_cm = pd.DataFrame(cm, 
                                 index=[f"Real: {status_aluno_map[i]}" for i in model_eval_instance.classes_], 
                                 columns=[f"Previsto: {status_aluno_map[i]}" for i in model_eval_instance.classes_])
            st.write("*Matriz de Confusão:*")
            st.table(df_cm)

            # Métricas de Desempenho
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, zero_division=0)
            recall = recall_score(y_test, y_pred_test, zero_division=0)
            f1 = f1_score(y_test, y_pred_test, zero_division=0)

            metrics_data = {
                'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
                'Valor': [accuracy, precision, recall, f1]
            }
            df_metrics = pd.DataFrame(metrics_data)
            st.write("*Métricas de Desempenho:*")
            st.table(df_metrics.style.format({'Valor': "{:.4f}"}))

        except Exception as e:
            st.error(f"Erro durante a avaliação do modelo {model_name}: {e}")
            continue # Pula para o próximo modelo se houver erro

        # 2. Previsão para o Aluno Inserido (treinando o modelo com TODOS os dados X_processed)
        st.write(f"**2. Previsão para o Aluno Inserido (usando {model_name} treinado com todos os dados):**")
        try:
            # Instanciar e treinar um novo modelo com todos os dados para a predição do input
            if model_name == 'Árvore de Decisão':
                model_for_input = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            else: # Random Forest
                model_for_input = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
            
            model_for_input.fit(X_processed, y_processed) # Treina com todos os dados processados
            
            prediction_input = model_for_input.predict(input_row_processed)
            prediction_proba_input = model_for_input.predict_proba(input_row_processed)

            # Nomes das classes para o DataFrame de probabilidades
            class_names_proba = [status_aluno_map[i] for i in model_for_input.classes_]
            df_prediction_proba_input = pd.DataFrame(prediction_proba_input, columns=class_names_proba)

            st.write("*Probabilidades de Evasão para o Aluno Inserido:*")
            st.dataframe(df_prediction_proba_input,
                         column_config={
                             status_aluno_map[0]: st.column_config.ProgressColumn(
                                 status_aluno_map[0],
                                 help=f"Probabilidade do aluno {status_aluno_map[0]}.",
                                 format="%.2f", # Probabilidade 0-1
                                 min_value=0,
                                 max_value=1,
                             ),
                             status_aluno_map[1]: st.column_config.ProgressColumn(
                                 status_aluno_map[1],
                                 help=f"Probabilidade do aluno {status_aluno_map[1]}.",
                                 format="%.2f", # Probabilidade 0-1
                                 min_value=0,
                                 max_value=1,
                             ),
                         }, hide_index=True, use_container_width=True)

            resultado_final_input = status_aluno_map[prediction_input[0]]
            if resultado_final_input == status_aluno_map[1]: # Evadirá
                st.error(f'**Predição Final ({model_name}): {resultado_final_input}** 😟')
                st.warning("Atenção: Este aluno apresenta risco de evasão segundo este modelo. Considere ações preventivas.")
            else: # Não Evadirá
                st.success(f'**Predição Final ({model_name}): {resultado_final_input}** 😊')
                            
            # Feature Importance (apenas para Random Forest, como exemplo)
            if model_name == 'Random Forest' and hasattr(model_for_input, 'feature_importances_'):
                with st.expander("💡 Importância das Features (Random Forest - treinado com todos os dados)"):
                    importances = model_for_input.feature_importances_
                    feature_names = X_processed.columns
                    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    df_importance = df_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                    st.bar_chart(df_importance.set_index('Feature'), use_container_width=True)

        except Exception as e:
            st.error(f"Erro durante a predição para o aluno inserido com o modelo {model_name}: {e}")
            # st.exception(e) # Para debugging mais detalhado

else:
    st.warning("O dataset não pôde ser carregado. O aplicativo não pode continuar.")

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido como exemplo de app de Machine Learning com Streamlit.")
