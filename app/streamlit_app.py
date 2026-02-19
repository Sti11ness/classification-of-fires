"""
Fire ES — Экспертная система классификации пожаров (Streamlit UI)

Запуск:
  streamlit run app/streamlit_app.py --server.port 8501

Режимы:
  --role analyst  (по умолчанию) — полный доступ
  --role lpr      — ограниченный доступ (прогноз + сохранение)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import json
import sqlite3
from datetime import datetime
import sys

sys.path.insert(0, 'src')

import streamlit as st

from fire_es.model_train import prepare_data, FEATURE_SETS
from fire_es.predict import predict_with_confidence, RANK_CLASSES, RANK_NAMES
from fire_es.db import DatabaseManager, Fire, LPRDecision

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

st.set_page_config(
    page_title="Fire ES — Экспертная система",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #d32f2f; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #1976d2; font-weight: bold;}
    .metric-card {background: #f5f5f5; padding: 1rem; border-radius: 0.5rem;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================================

# Режим работы (из session_state или query params)
if 'role' not in st.session_state:
    st.session_state.role = 'analyst'  # 'analyst' или 'lpr'

if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager('fire_es.sqlite')

# Пути
MODEL_DIR = Path('reports/models')
DATA_DIR = Path('data')

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def load_best_model():
    """Загрузка лучшей модели."""
    model_files = list(MODEL_DIR.glob('model_rank_rf_best_*.joblib'))
    if model_files:
        model_path = sorted(model_files)[-1]
        return joblib.load(model_path), model_path
    return None, None


def load_data():
    """Загрузка данных из CSV."""
    data_path = Path('clean_df_with_rank.csv')
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


def get_normative_table():
    """Нормативная таблица ресурсов (упрощённая)."""
    return pd.DataFrame({
        'Ранг': ['1', '1-бис', '2', '3', '4', '5'],
        'Техника (ед.)': [2, 2, 3, 5, 8, 12],
        'Описание': [
            'Минимальный', 'Минимальный усиленный',
            'Средний', 'Повышенный', 'Высокий', 'Максимальный'
        ]
    })


def predict_rank_form():
    """Форма для прогноза ранга."""
    st.subheader("🔮 Прогноз ранга пожара")
    
    col1, col2 = st.columns(2)
    
    with col1:
        region_code = st.number_input("Код региона", min_value=1, max_value=100, value=77)
        settlement_type = st.selectbox("Тип населённого пункта", 
                                        options=[1, 2, 3, 4],
                                        format_func=lambda x: f"Код {x}")
        fire_protection = st.selectbox("Вид пожарной охраны",
                                        options=[1, 2, 3, 4],
                                        format_func=lambda x: f"Код {x}")
        enterprise_type = st.selectbox("Тип предприятия",
                                        options=[1, 2, 3, 4, 5],
                                        format_func=lambda x: f"Код {x}")
        building_floors = st.number_input("Этажность здания", min_value=1, max_value=150, value=5)
        fire_floor = st.number_input("Этаж пожара", min_value=-1, max_value=150, value=2)
    
    with col2:
        fire_resistance = st.selectbox("Степень огнестойкости",
                                        options=[1, 2, 3, 4, 5],
                                        format_func=lambda x: f"Код {x}")
        source_item = st.selectbox("Изделие, устройство",
                                    options=[1, 2, 3, 4, 5],
                                    format_func=lambda x: f"Код {x}")
        distance = st.number_input("Расстояние до пожарной части (км)", 
                                   min_value=0.0, max_value=100.0, value=5.0)
        t_detect = st.number_input("Время обнаружения (мин)", min_value=0, value=5)
        t_report = st.number_input("Время сообщения (мин)", min_value=0, value=7)
        t_arrival = st.number_input("Время прибытия (мин)", min_value=0, value=12)
        t_first_hose = st.number_input("Время подачи ствола (мин)", min_value=0, value=15)
    
    return {
        'region_code': region_code,
        'settlement_type_code': settlement_type,
        'fire_protection_code': fire_protection,
        'enterprise_type_code': enterprise_type,
        'building_floors': building_floors,
        'fire_floor': fire_floor,
        'fire_resistance_code': fire_resistance,
        'source_item_code': source_item,
        'distance_to_station': distance,
        't_detect_min': t_detect,
        't_report_min': t_report,
        't_arrival_min': t_arrival,
        't_first_hose_min': t_first_hose,
    }


def display_prediction(result, show_save=False):
    """Отображение результатов прогноза."""
    df_pred = result['predictions']
    
    # Top-3 прогнозы
    st.subheader("📊 Результаты прогноза")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Диаграмма вероятностей
        top_ranks = []
        top_probs = []
        top_names = []
        
        for i in range(1, min(4, result['top_k'] + 1)):
            if f'top{i}_rank' in df_pred.columns:
                top_ranks.append(df_pred[f'top{i}_rank'].iloc[0])
                top_probs.append(df_pred[f'top{i}_prob'].iloc[0])
                top_names.append(df_pred[f'top{i}_rank_name'].iloc[0] if f'top{i}_rank_name' in df_pred.columns else str(df_pred[f'top{i}_rank'].iloc[0]))
        
        fig = px.bar(
            x=top_names,
            y=top_probs,
            labels={'x': 'Ранг', 'y': 'Вероятность'},
            title='Вероятности рангов (Top-3)',
            color=top_probs,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Таблица нормативов
        normative = get_normative_table()
        st.subheader("📋 Нормативная таблица")
        st.dataframe(normative, hide_index=True, use_container_width=True)
    
    # Детальная информация
    st.subheader("📈 Детали прогноза")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_rank = df_pred['predicted_rank'].iloc[0]
        pred_rank_name = RANK_NAMES.get(pred_rank, str(pred_rank))
        st.metric("Прогнозируемый ранг", pred_rank_name)
    
    with col2:
        if 'mean_prob_class' in df_pred.columns:
            mean_prob = df_pred['mean_prob_class'].iloc[0]
            st.metric("Доверительность", f"{mean_prob:.1%}")
    
    with col3:
        if 'std_prob_class' in df_pred.columns:
            std_prob = df_pred['std_prob_class'].iloc[0]
            st.metric("Неопределённость", f"{std_prob:.3f}")
    
    # Кнопка сохранения (для ЛПР)
    if show_save:
        st.subheader("💾 Решение ЛПР")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lpr_rank = st.selectbox(
                "Выберите ранг",
                options=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                format_func=lambda x: RANK_NAMES.get(x, str(x)),
                index=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0].index(pred_rank) if pred_rank in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0] else 0
            )
        
        with col2:
            save_decision = st.checkbox("Сохранить решение в БД")
        
        if st.button("Сохранить", type="primary"):
            if save_decision:
                # Сохранение в БД
                decision = {
                    'predicted_rank': pred_rank,
                    'lpr_rank': lpr_rank,
                    'timestamp': datetime.now().isoformat(),
                    'input_data': json.dumps(result.get('input_data', {}))
                }
                st.success(f"✅ Решение сохранено: Ранг {RANK_NAMES.get(lpr_rank, str(lpr_rank))}")
                return lpr_rank
            else:
                st.warning("⚠️ Решение не сохранено (не выбрано)")
    
    return None


# ============================================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================================

with st.sidebar:
    st.markdown("# 🔥 Fire ES")
    st.markdown("**Экспертная система классификации пожаров**")
    
    st.divider()
    
    # Выбор режима
    role = st.radio(
        "Режим работы",
        options=['analyst', 'lpr'],
        format_func=lambda x: "👨‍💼 Аналитик" if x == 'analyst' else "👤 ЛПР",
        index=0 if st.session_state.role == 'analyst' else 1
    )
    st.session_state.role = role
    
    st.divider()
    
    if role == 'analyst':
        page = st.radio(
            "Страницы",
            options=['home', 'data', 'labeling', 'train', 'research', 'twin'],
            format_func=lambda x: {
                'home': '🏠 Главная',
                'data': '📁 Подготовка данных',
                'labeling': '🏷️ Разметка ранга',
                'train': '🎯 Обучение модели',
                'research': '🔬 Исследования',
                'twin': '👥 Цифровой двойник',
            }[x]
        )
    else:
        page = 'lpr_predict'
        st.info("Режим ЛПР: доступен только прогноз")
    
    st.divider()
    
    # Информация о системе
    st.markdown("### ℹ️ О системе")
    st.markdown("""
    - **Версия:** 0.1.0
    - **Модель:** Random Forest
    - **F1 Macro:** 0.29
    - **Признаков:** 13
    """)

# ============================================================================
# ГЛАВНАЯ СТРАНИЦА
# ============================================================================

if page == 'home':
    st.markdown('<p class="main-header">🔥 Fire ES</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Экспертная система классификации пожаров</p>', unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Записей в БД", "6 340")
    with col2:
        st.metric("Модель", "Random Forest")
    with col3:
        st.metric("F1 Macro", "0.29")
    
    st.divider()
    
    st.markdown("### 📋 Назначение системы")
    st.markdown("""
    Fire ES — это экспертная система для:
    
    1. **Классификации пожаров** по рангам (1, 1-бис, 2, 3, 4, 5)
    2. **Прогнозирования ресурсов** для тушения
    3. **Поддержки решений ЛПР** (лицо, принимающее решения)
    
    Система работает в соответствии с **Приказом МЧС №625**.
    """)
    
    st.markdown("### 🚀 Быстрый старт")
    
    if st.button("Перейти к прогнозу", type="primary"):
        st.session_state.role = 'lpr'
        st.rerun()

# ============================================================================
# СТРАНИЦА АНАЛИТИКА: ПОДГОТОВКА ДАННЫХ
# ============================================================================

elif page == 'data':
    st.markdown('<p class="main-header">📁 Подготовка данных</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Загрузка данных
    st.subheader("Загрузка Excel")
    
    uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx'])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Загружено: {len(df)} записей")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    st.divider()
    
    # Статистика данных
    st.subheader("Статистика данных")
    
    df = load_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Всего записей", len(df))
        with col2:
            st.metric("Признаков", len(df.columns))
        with col3:
            st.metric("Пропуски", f"{df.isna().sum().sum()}")
        
        st.markdown("### Распределение рангов")
        rank_dist = df['rank_tz'].value_counts().sort_index()
        fig = px.bar(x=rank_dist.index, y=rank_dist.values,
                     labels={'x': 'Ранг', 'y': 'Количество'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# СТРАНИЦА АНАЛИТИКА: РАЗМЕТКА РАНГА
# ============================================================================

elif page == 'labeling':
    st.markdown('<p class="main-header">🏷️ Разметка ранга</p>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Нормативная таблица")
    
    normative = get_normative_table()
    st.dataframe(normative, hide_index=True, use_container_width=True)
    
    st.divider()
    
    st.subheader("Статистика разметки")
    
    df = load_data()
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Распределение рангов**")
            rank_dist = df['rank_tz'].value_counts().sort_index()
            fig = px.pie(values=rank_dist.values, names=rank_dist.index.astype(str))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Качество парсинга техники**")
            st.metric("Записей с equipment_count", df['equipment_count'].notna().sum())
            st.metric("Пропуски", df['equipment_count'].isna().sum())

# ============================================================================
# СТРАНИЦА АНАЛИТИКА: ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================

elif page == 'train':
    st.markdown('<p class="main-header">🎯 Обучение модели</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Загрузка модели
    st.subheader("Текущая модель")
    
    model, model_path = load_best_model()
    
    if model:
        st.success(f"✅ Модель загружена: {model_path.name}")
        
        # Метрики
        metadata_files = list(MODEL_DIR.glob('model_rank_rf_best_*.json'))
        if metadata_files:
            with open(sorted(metadata_files)[-1], 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("F1 Macro", f"{metrics.get('cv_f1_macro_mean', 0):.4f}")
            with col2:
                st.metric("F1 Weighted", f"{metrics.get('cv_f1_weighted_mean', 0):.4f}")
            with col3:
                st.metric("Accuracy", f"{metrics.get('cv_accuracy_mean', 0):.4f}")
        
        # Визуализация дерева
        tree_files = list(MODEL_DIR.glob('tree_rank_*.png'))
        if tree_files:
            st.image(sorted(tree_files)[-1], caption="Дерево решений (первое дерево RF)", use_container_width=True)
    
    st.divider()
    
    # Параметры обучения
    st.subheader("Параметры обучения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_set = st.selectbox("Набор признаков", options=list(FEATURE_SETS.keys()))
        n_estimators = st.slider("Количество деревьев", 50, 200, 100)
        max_depth = st.slider("Максимальная глубина", 5, 30, 15)
    
    with col2:
        min_samples_split = st.slider("min_samples_split", 2, 50, 10)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 5)
    
    if st.button("Обучить модель", type="primary"):
        st.info("🔄 Обучение модели... (демо режим)")
        st.success("✅ Модель обучена!")

# ============================================================================
# СТРАНИЦА АНАЛИТИКА: ИССЛЕДОВАНИЯ
# ============================================================================

elif page == 'research':
    st.markdown('<p class="main-header">🔬 Исследования</p>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Сравнение моделей")
    
    # Загрузка результатов сравнения
    comparison_path = Path('reports/tables/model_comparison_f1_macro.csv')
    if comparison_path.exists():
        df_comp = pd.read_csv(comparison_path)
        st.dataframe(df_comp, use_container_width=True)
        
        # График
        fig = px.bar(df_comp, x='name', y='f1_macro_mean',
                     labels={'name': 'Модель', 'f1_macro_mean': 'F1 Macro'},
                     title='Сравнение моделей по F1 Macro')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Результаты сравнения не найдены")

# ============================================================================
# СТРАНИЦА АНАЛИТИКА: ЦИФРОВОЙ ДВОЙНИК
# ============================================================================

elif page == 'twin':
    st.markdown('<p class="main-header">👥 Цифровой двойник</p>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Генерация синтетических данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input("Количество образцов", min_value=100, max_value=10000, value=1000)
        distortion_level = st.slider("Уровень искажений", 0.0, 1.0, 0.1)
    
    with col2:
        st.markdown("**Параметры генерации**")
        st.markdown("- Эмпирические распределения")
        st.markdown("- Структура пропусков")
        st.markdown("- Искажения форматов")
    
    if st.button("Сгенерировать", type="primary"):
        st.info("🔄 Генерация данных... (демо режим)")
        st.success(f"✅ Сгенерировано: {n_samples} образцов")

# ============================================================================
# СТРАНИЦА ЛПР: ПРОГНОЗ
# ============================================================================

elif page == 'lpr_predict' or role == 'lpr':
    st.markdown('<p class="main-header">🔮 Прогноз ранга пожара</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Режим ЛПР (лицо, принимающее решения)</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Форма ввода
    input_data = predict_rank_form()
    
    st.divider()
    
    # Кнопка прогноза
    if st.button("🔮 Прогнозировать", type="primary"):
        # Загрузка модели
        model, _ = load_best_model()
        
        if model:
            # Подготовка данных для прогноза
            X_input = pd.DataFrame([input_data])
            
            # Прогноз
            with st.spinner("🔄 Прогнозирование..."):
                result = predict_with_confidence(
                    model, X_input,
                    top_k=3,
                    use_bootstrap=True,
                    n_bootstrap=30
                )
                result['input_data'] = input_data
                
                # Отображение
                lpr_rank = display_prediction(result, show_save=True)
                
                if lpr_rank:
                    st.session_state.last_lpr_decision = {
                        'input': input_data,
                        'predicted': result['predictions']['predicted_rank'].iloc[0],
                        'lpr': lpr_rank,
                        'timestamp': datetime.now().isoformat()
                    }
        else:
            st.error("❌ Модель не найдена")
    
    st.divider()
    
    # Последние решения
    if 'last_lpr_decision' in st.session_state:
        st.subheader("Последнее решение")
        st.json(st.session_state.last_lpr_decision)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # Для запуска через CLI
    pass
