import os
import json
import numpy as np

from mylib import myDataPrep as dp
from mylib import mySentimentModel as sm

# ---- 경로 / 설정 ----
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'combined_reviews.csv')
MODEL_DIR = os.path.join(BASE, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'sa_model_en.keras')
TOKENIZER_FILE = os.path.join(MODEL_DIR, 'sa_tokenizer_en.pkl')
META_FILE = os.path.join(MODEL_DIR, 'meta.json')

MAX_ROWS = 200000     # 대용량(13GB) 중 앞부분만 사용 (빠른 데모)
PER_CLASS = 12000     # 클래스당 표본 수
VOCAB_SIZE = sm.VOCAB_SIZE
MAX_LEN = sm.MAX_LEN
EPOCHS = 8
BATCH_SIZE = 128


def run(progress_cb=None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ===== 1. 데이터 준비 =====
    print('[1] 데이터 준비 ...')
    df = dp.load_balanced_english(DATA_PATH, max_rows=MAX_ROWS, per_class=PER_CLASS)
    print(f'    영어 균형 데이터셋: {len(df):,}건 (긍정 {int(df.label.sum()):,} / 부정 {int((1-df.label).sum()):,})')

    from sklearn.model_selection import train_test_split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df['clean_text'].tolist(), df['label'].tolist(),
        test_size=0.1, stratify=df['label'], random_state=42)

    # Integer Encoding + Padding + One-hot
    tokenizer = dp.build_tokenizer(X_train_txt, vocab_size=VOCAB_SIZE)
    train_X = dp.encode_pad(tokenizer, X_train_txt, max_len=MAX_LEN)
    test_X = dp.encode_pad(tokenizer, X_test_txt, max_len=MAX_LEN)

    from tensorflow.keras.utils import to_categorical
    train_y = to_categorical(y_train, 2)
    test_y = to_categorical(y_test, 2)

    # ===== 2. 모델 구축 & 컴파일 =====
    print('[2] 모델 구축 & 컴파일 ...')
    model = sm.build_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    model.summary()

    # ===== 3. 모델 학습 =====
    print('[3] 모델 학습 ...')
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3,
                       restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(MODEL_FILE, monitor='val_loss', mode='min',
                         save_best_only=True)
    callbacks = [es, mc]
    if progress_cb is not None:
        callbacks.append(progress_cb)

    history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=callbacks, verbose=2)

    # ===== 4. 성능 평가 =====
    print('[4] 성능 평가 ...')
    loss, acc = model.evaluate(test_X, test_y, verbose=0)
    preds = model.predict(test_X, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, y_pred, target_names=['부정', '긍정'],
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f'    test loss={loss:.4f}  acc={acc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['부정', '긍정']))

    # ===== 5. 배포 (저장) =====
    print('[5] 배포(저장) ...')
    model.save(MODEL_FILE)
    import joblib
    joblib.dump(tokenizer, TOKENIZER_FILE)
    meta = {
        'max_len': MAX_LEN,
        'vocab_size': VOCAB_SIZE,
        'n_samples': len(df),
        'test_loss': float(loss),
        'test_acc': float(acc),
        'report': report,
        'confusion_matrix': cm,
        'history': {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'    저장 완료: {MODEL_FILE}')
    return meta


if __name__ == '__main__':
    run()
