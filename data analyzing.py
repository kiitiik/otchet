import pandas as pd
import re
from collections import Counter

# Загрузка данных
data = pd.read_excel('очищенный_файл.xlsx')

# Очистка текста
def clean_text(text):
    text = re.sub(r'&nbsp;', ' ', text)  # Удаляем HTML-код
    text = re.sub(r'[\\.,!?:;"()\[\]{}]', '', text)  # Убираем знаки препинания
    text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
    return text.lower().strip()

data['ParamValue_cleaned'] = data['ParamValue'].apply(clean_text)

# Расширение правил разметки
serious_keywords = ['ударил', 'травма', 'перелом', 'ожог', 'кровь']
mild_keywords = ['кашель', 'насморк', 'боль в горле', 'боль', 'температура', 'простуда']
norma_keywords = ['чувствую хорошо', 'норма', 'здоров']

def label_text(text):
    if any(word in text for word in serious_keywords):
        return 'серьёзное'
    elif any(word in text for word in mild_keywords):
        return 'лёгкое'
    elif any(word in text for word in norma_keywords):
        return 'норма'
    else:
        return 'неопределено'

data['Sentiment'] = data['ParamValue_cleaned'].apply(label_text)

# Проверка сбалансированности разметки
label_counts = Counter(data['Sentiment'])
print("Распределение меток:\n", label_counts)

# Удаление дубликатов
data = data.drop_duplicates(subset=['ParamValue_cleaned'], keep='first')

# Сохранение результата разметки
data.to_excel('labeled_data.xlsx', index=False)
print("Файл с разметкой сохранён как 'labeled_data.xlsx'.")

# Анализ категорий и ключевых слов
def analyze_keywords(data, keywords, label):
    matched_data = data[data['Sentiment'] == label]
    keyword_counts = {word: matched_data['ParamValue_cleaned'].str.contains(word).sum() for word in keywords}
    return keyword_counts

serious_analysis = analyze_keywords(data, serious_keywords, 'серьёзное')
mild_analysis = analyze_keywords(data, mild_keywords, 'лёгкое')
norma_analysis = analyze_keywords(data, norma_keywords, 'норма')

# Создание отчёта анализа
report = pd.DataFrame({
    'Серьёзное': serious_analysis,
    'Лёгкое': mild_analysis,
    'Норма': norma_analysis
}).fillna(0)

report.to_excel('keywords_analysis.xlsx', index=True)
print("Отчёт о ключевых словах сохранён как 'keywords_analysis.xlsx'.")
