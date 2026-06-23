"""
AI 서비스 개발 통합 웹 대시보드 (Streamlit)

강의 03(Streamlit) 미니프로젝트 + 강의 09(순환신경망 감성분석) 을 하나의 앱으로 통합.

프로젝트 구성 (강의 03 패키지 구조 적용)
  감성분석/
    data/   combined_reviews.csv
    mylib/  myTextAnalyzer.py        # (PDF) 단어 빈도수 분석 로직
            myStreamlitVisualizer.py # (PDF) Streamlit 시각화 (막대/워드클라우드)
            myDataPrep.py            # 감성분석 데이터 준비
            mySentimentModel.py      # 감성분석 모델/배포
    model/  sa_model_en.keras, sa_tokenizer_en.pkl, meta.json
    train_pipeline.py
    SentimentWebDashboard.py   <- 본 파일 (메인 대시보드)

실행:
  conda activate aiservice26
  streamlit run SentimentWebDashboard.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from mylib import myDataPrep as dp
from mylib import mySentimentModel as sm
from mylib import myTextAnalyzer as ta
from mylib import myStreamlitVisualizer as viz
import train_pipeline as tp

# ---- 한글 폰트 ----
FONT_PATH = r'C:\Windows\Fonts\malgun.ttf'
if os.path.exists(FONT_PATH):
    rc('font', family=font_manager.FontProperties(fname=FONT_PATH).get_name())
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title='AI 서비스 개발 대시보드', page_icon='🧭', layout='wide')

# 기본 한국어 불용어 (조사/접속 등)
DEFAULT_STOPWORDS = ['은', '는', '이', '가', '을', '를', '에', '의', '도', '으로',
                     '와', '과', '한', '하다', '있다', '없다', '되다', '것', '들',
                     '그', '저', '수', '등', '더', '점']


# =====================================================================
# 공통 캐시 (감성분석)
# =====================================================================
@st.cache_data(show_spinner='데이터 로딩 중...')
def load_sample(max_rows=200000, per_class=12000):
    return dp.load_balanced_english(tp.DATA_PATH, max_rows=max_rows, per_class=per_class)


def model_ready():
    return os.path.exists(tp.MODEL_FILE) and os.path.exists(tp.TOKENIZER_FILE)


@st.cache_resource(show_spinner='모델 로딩 중...')
def get_analyzer():
    max_len = sm.MAX_LEN
    if os.path.exists(tp.META_FILE):
        with open(tp.META_FILE, encoding='utf-8') as f:
            max_len = json.load(f).get('max_len', sm.MAX_LEN)
    return sm.SentimentAnalyzer(tp.MODEL_FILE, tp.TOKENIZER_FILE, max_len=max_len)


def load_meta():
    if os.path.exists(tp.META_FILE):
        with open(tp.META_FILE, encoding='utf-8') as f:
            return json.load(f)
    return None


# =====================================================================
# 공통 캐시 (단어 빈도수 분석)
# =====================================================================
@st.cache_resource(show_spinner='형태소 분석기 로딩 중...')
def get_okt():
    from konlpy.tag import Okt
    return Okt()


@st.cache_data(show_spinner='형태소 분석 중...')
def run_word_freq(corpus, tags, stopwords):
    """corpus(tuple) 를 토큰화하여 Counter 반환 (캐시 가능 형태)."""
    okt = get_okt()
    tokens = ta.tokenize_korean_corpus(list(corpus), okt,
                                       my_tags=list(tags) if tags else None,
                                       my_stopwords=list(stopwords) if stopwords else None)
    return ta.analyze_word_freq(tokens)


# =====================================================================
# 사이드바 (앱 선택)
# =====================================================================
st.sidebar.title('🧭 AI 서비스 개발')
app = st.sidebar.radio('앱 선택',
                       ['📝 단어 빈도수 분석 (강의 03)',
                        '🎬 감성분석 대시보드 (강의 09)'])


# #####################################################################
# 앱 A. 단어 빈도수 분석 웹 대시보드  (강의 03 PDF 미니프로젝트)
# #####################################################################
if app.startswith('📝'):
    st.sidebar.markdown('---')
    st.sidebar.subheader('⚙️ 분석 옵션')
    tag_map = {'명사': 'Noun', '동사': 'Verb', '형용사': 'Adjective', '부사': 'Adverb'}
    sel_tags_ko = st.sidebar.multiselect('추출할 품사', list(tag_map.keys()),
                                         default=['명사'])
    sel_tags = tuple(tag_map[k] for k in sel_tags_ko)
    num_words = st.sidebar.slider('표시할 단어 수', 5, 100, 30)
    stop_text = st.sidebar.text_area('불용어 (쉼표/줄바꿈 구분)',
                                     ', '.join(DEFAULT_STOPWORDS), height=100)
    stopwords = tuple(w.strip() for w in stop_text.replace('\n', ',').split(',') if w.strip())

    st.title('📝 단어 빈도수 분석 웹 대시보드')
    st.caption('강의 03 Streamlit 미니프로젝트 — 한국어 형태소 분석(Okt) 기반 단어 빈도/워드클라우드')

    with st.expander('ℹ️ Streamlit 기본 API (PDF 내용)'):
        st.markdown('**1) Write / Magic Command** — `st.write()` 없이 변수만 적어도 자동 출력')
        '👉 이 문장은 magic command 로 출력된 것입니다.'
        st.markdown('**2) Text 출력** — `st.title / st.header / st.subheader / st.markdown / st.code`')
        st.code("import streamlit as st\nst.write('Hello, Streamlit!')", language='python')
        st.markdown('**3) Input widgets** — text_input, slider, selectbox, multiselect, file_uploader …')
        st.markdown('**4) Form** — 여러 항목을 입력 후 submit 시 한꺼번에 처리')
        st.markdown('**5) sidebar / @st.cache_data** — 옵션은 왼쪽 사이드바, 무거운 연산은 캐싱')

    # ---- 데이터 입력 (form) ----
    st.subheader('1) 데이터 입력')
    source = st.radio('입력 방식', ['직접 입력', 'CSV 업로드'], horizontal=True)

    corpus = None
    if source == '직접 입력':
        with st.form('text_form'):
            text = st.text_area('분석할 텍스트 (한 줄에 한 문장 권장)', height=180,
                                value='이 영화 정말 재미있고 감동적이었어요\n'
                                      '배우들의 연기가 훌륭했습니다\n'
                                      '스토리가 탄탄하고 연출도 좋았다\n'
                                      '음악과 영상미가 인상적인 작품')
            submitted = st.form_submit_button('분석하기', type='primary')
        if submitted and text.strip():
            corpus = tuple(line for line in text.splitlines() if line.strip())
    else:
        with st.form('csv_form'):
            uploaded = st.file_uploader('CSV 파일 업로드', type='csv')
            max_rows = st.slider('읽을 행 수 (앞부분)', 100, 5000, 1000, step=100)
            col_name = st.text_input('분석할 열 이름', 'document')
            submitted = st.form_submit_button('분석하기', type='primary')
        if submitted and uploaded is not None:
            df_in = pd.read_csv(uploaded, nrows=max_rows)
            if col_name not in df_in.columns:
                st.error(f"'{col_name}' 열이 없습니다. 사용 가능한 열: {list(df_in.columns)}")
            else:
                df_in = df_in.dropna(subset=[col_name])
                corpus = tuple(df_in[col_name].astype(str).tolist())
                st.dataframe(df_in.head(), use_container_width=True)

    # ---- 분석 결과 ----
    if corpus:
        counter = run_word_freq(corpus, sel_tags, stopwords)
        if not counter:
            st.warning('추출된 단어가 없습니다. 품사/불용어 옵션을 확인하세요.')
        else:
            st.subheader('2) 분석 결과')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('**📊 빈도 막대그래프**')
                fig_bar = viz.visualize_barhgraph(counter, num_words,
                                                  xlabel='빈도', font_path=FONT_PATH)
                st.pyplot(fig_bar)
            with col2:
                st.markdown('**☁️ 워드클라우드**')
                fig_wc = viz.visualize_wordcloud(counter, num_words, FONT_PATH)
                st.pyplot(fig_wc)

            st.markdown('**📋 상위 단어 표**')
            top = counter.most_common(num_words)
            st.dataframe(pd.DataFrame(top, columns=['단어', '빈도']),
                         use_container_width=True)
    else:
        st.info('텍스트를 입력하거나 CSV 를 업로드한 뒤 "분석하기" 를 누르세요.')


# #####################################################################
# 앱 B. 감성분석 대시보드  (강의 09)
# #####################################################################
else:
    st.sidebar.markdown('---')
    st.sidebar.markdown('### 🎬 감성분석 대시보드')
    page = st.sidebar.radio('단계 선택',
                            ['📊 1. 데이터 준비',
                             '🧠 2~4. 모델 구축·학습·평가',
                             '💬 5~6. 감정 분석 (배포)'])
    st.sidebar.caption('환경: aiservice26 (TF 2.20)\n전략: 영어 전용 단일 LSTM')

    # ---------- 페이지 1. 데이터 준비 ----------
    if page.startswith('📊'):
        st.title('📊 1. 데이터 준비')
        st.subheader('혼합언어 문제와 해결 전략')
        st.markdown(
            """
**문제** — `combined_reviews.csv` 는 본문의 **약 99.9% 가 영어/다국어**, 한국어는 **0.11%** 에
불과한 혼합 데이터입니다. 강의 방식(Okt + Keras Tokenizer + LSTM)은 단어 하나하나를
인덱스로 외우므로, 한국어 `최고` 와 영어 `great` 는 완전히 다른 토큰입니다.
같이 넣으면 **어휘만 2배로 불어나고 전이가 일어나지 않아** 정확도가 떨어집니다.

**해결** — 입력 단계에서 **영어 리뷰만 선별**해 단일 언어로 깨끗한 어휘를 구성합니다
(한글/키릴/일본어/한자 포함 행 제외). 평점은 라벨이 없으므로 **≥7 → 긍정, ≤4 → 부정**
으로 변환하고 5~6점·N/A 는 제외합니다.
            """
        )
        df = load_sample()
        c1, c2, c3 = st.columns(3)
        c1.metric('표본 수', f'{len(df):,}')
        c2.metric('긍정(≥7)', f'{int(df.label.sum()):,}')
        c3.metric('부정(≤4)', f'{int((1 - df.label).sum()):,}')

        st.subheader('샘플 미리보기')
        show = df[['Review_Rating', 'label', 'Review_Text']].head(15).copy()
        show['label'] = show['label'].map({1: '긍정', 0: '부정'})
        st.dataframe(show, use_container_width=True)

        colL, colR = st.columns(2)
        with colL:
            st.subheader('라벨 분포')
            fig, ax = plt.subplots(figsize=(4, 3))
            counts = df.label.map({1: '긍정', 0: '부정'}).value_counts()
            ax.bar(counts.index, counts.values, color=['#4C9F70', '#D9534F'])
            ax.set_ylabel('리뷰 수')
            st.pyplot(fig)
        with colR:
            st.subheader('평점 분포')
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            df['Review_Rating'].plot(kind='hist', bins=10, ax=ax2, color='#5B8DEF')
            ax2.set_xlabel('Review_Rating')
            st.pyplot(fig2)

    # ---------- 페이지 2. 모델 구축·학습·평가 ----------
    elif page.startswith('🧠'):
        st.title('🧠 2~4. 모델 구축 · 학습 · 평가')
        st.markdown(
            f"""
**모델 구조 (강의와 동일)** — `Embedding({sm.VOCAB_SIZE + 1}, {sm.EMBEDDING_DIM})`
→ `LSTM({sm.LSTM_UNITS})` → `Dense({sm.DENSE_UNITS}, tanh)` → `Dense(2, softmax)`
· 손실 `binary_crossentropy` · 최적화 `RMSprop` · `max_len={sm.MAX_LEN}`
            """
        )
        if st.button('🚀 학습 실행 (데이터준비 → 구축 → 학습 → 평가 → 저장)', type='primary'):
            prog = st.progress(0.0, text='학습 준비 중...')
            status = st.empty()
            from tensorflow.keras.callbacks import Callback

            class StProgress(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    frac = (epoch + 1) / tp.EPOCHS
                    prog.progress(min(frac, 1.0), text=f'Epoch {epoch + 1}/{tp.EPOCHS}')
                    status.write(
                        f"epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
                        f"acc={logs.get('accuracy', 0):.4f}, "
                        f"val_loss={logs.get('val_loss', 0):.4f}, "
                        f"val_acc={logs.get('val_accuracy', 0):.4f}")

            with st.spinner('학습 중... (수 분 소요)'):
                tp.run(progress_cb=StProgress())
            prog.progress(1.0, text='완료')
            get_analyzer.clear()
            st.success('학습 및 저장 완료!')

        meta = load_meta()
        if not meta:
            st.info('아직 학습된 모델이 없습니다. 위 버튼으로 학습을 실행하세요.')
        else:
            st.subheader('성능 평가')
            c1, c2 = st.columns(2)
            c1.metric('테스트 정확도', f"{meta['test_acc'] * 100:.2f}%")
            c2.metric('테스트 손실', f"{meta['test_loss']:.4f}")

            colL, colR = st.columns(2)
            with colL:
                st.markdown('**학습 곡선**')
                h = meta['history']
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(h['accuracy'], label='train acc')
                if 'val_accuracy' in h:
                    ax.plot(h['val_accuracy'], label='val acc')
                ax.set_xlabel('epoch'); ax.set_ylabel('accuracy'); ax.legend()
                st.pyplot(fig)
            with colR:
                st.markdown('**혼동 행렬 (Confusion Matrix)**')
                cm = np.array(meta['confusion_matrix'])
                fig2, ax2 = plt.subplots(figsize=(4, 3.5))
                ax2.imshow(cm, cmap='Blues')
                ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
                ax2.set_xticklabels(['부정', '긍정']); ax2.set_yticklabels(['부정', '긍정'])
                ax2.set_xlabel('예측'); ax2.set_ylabel('실제')
                for i in range(2):
                    for j in range(2):
                        ax2.text(j, i, cm[i, j], ha='center', va='center',
                                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
                st.pyplot(fig2)

            st.markdown('**분류 리포트**')
            rep = meta['report']
            rows = [{'클래스': k, 'precision': rep[k]['precision'],
                     'recall': rep[k]['recall'], 'f1-score': rep[k]['f1-score'],
                     'support': rep[k]['support']} for k in ['부정', '긍정']]
            st.dataframe(pd.DataFrame(rows).set_index('클래스').round(3),
                         use_container_width=True)

    # ---------- 페이지 3. 감정 분석 (배포) ----------
    else:
        st.title('💬 5~6. 감정 분석 (배포)')
        if not model_ready():
            st.warning('학습된 모델이 없습니다. "2~4. 모델 학습" 페이지에서 먼저 학습하세요.')
            st.stop()

        sa = get_analyzer()
        st.caption('영어 영화 리뷰를 입력하면 긍/부정을 판단합니다.')
        text = st.text_area('리뷰 입력 (English)',
                            'This movie was an absolute masterpiece, I loved every minute of it!',
                            height=120)
        if st.button('분석하기', type='primary') and text.strip():
            label, prob, probs = sa.analyze(text)
            emoji = '😊' if label == '긍정' else '😞'
            st.markdown(f"## {emoji} **{label}**  ({prob * 100:.2f}%)")
            st.progress(probs['긍정'],
                        text=f"긍정 {probs['긍정']*100:.1f}%  /  부정 {probs['부정']*100:.1f}%")

        st.divider()
        st.subheader('예시 리뷰 일괄 분석')
        examples = [
            'What a masterpiece, absolutely loved it!',
            'Terrible film, complete waste of time.',
            'The acting was brilliant and the story was touching.',
            'Boring and predictable, I fell asleep.',
            'One of the best movies I have ever seen.',
            'Worst movie ever, do not watch this garbage.',
        ]
        if st.button('예시 분석 실행'):
            out = []
            for ex in examples:
                label, prob, _ = sa.analyze(ex)
                out.append({'리뷰': ex, '예측': label, '확률(%)': round(prob * 100, 2)})
            st.dataframe(pd.DataFrame(out), use_container_width=True)
