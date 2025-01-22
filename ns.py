import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
# Загрузка данных
data = pd.read_excel('labeled_data.xlsx')

# Настройки для обработки текста
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('russian'))  # Убедитесь, что установлен пакет nltk

# Предобработка текста
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Удаление цифр
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()

data['ParamValue_cleaned'] = data['ParamValue_cleaned'].apply(preprocess_text)

# Преобразование меток в числовой формат
label_mapping = {'норма': 0, 'лёгкое': 1, 'серьёзное': 2, 'неопределено': 3}
data['Sentiment'] = data['Sentiment'].map(label_mapping)

# Удаление записей с неопределённой меткой
data = data[data['Sentiment'] != 3]

# Токенизация
max_length = 50
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
data['tokens'] = data['ParamValue_cleaned'].apply(
    lambda x: tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True
    )
)

# Разделение данных
X = list(data['tokens'])
y = data['Sentiment']

train_data, temp_data = train_test_split(data, test_size=0.3, stratify=y, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['Sentiment'], random_state=42)

# Подготовка данных для модели
train_dataset = TensorDataset(
    torch.tensor([item['input_ids'] for item in train_data['tokens']], dtype=torch.long),
    torch.tensor([item['attention_mask'] for item in train_data['tokens']], dtype=torch.long),
    torch.tensor(train_data['Sentiment'].values, dtype=torch.long)
)

validation_dataset = TensorDataset(
    torch.tensor([item['input_ids'] for item in validation_data['tokens']], dtype=torch.long),
    torch.tensor([item['attention_mask'] for item in validation_data['tokens']], dtype=torch.long),
    torch.tensor(validation_data['Sentiment'].values, dtype=torch.long)
)

# Вычисление весов классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),  #
    y=train_data['Sentiment']
)

weights = torch.tensor(class_weights, dtype=torch.float)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

# Настройка DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16)

# Настройка модели
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Обучение модели
model.train()
num_epochs = 3    
loss_values = []

for epoch in range(num_epochs):
    epoch_loss = 0
    print(f"\nНачинается эпоха {epoch + 1}...\n")

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100, unit='batch'):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_epoch_loss)
    print(f"Средняя потеря за эпоху {epoch + 1}: {avg_epoch_loss}")

# Валидация модели
model.eval()
predictions = []
true_labels = []

for batch in validation_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

# Метрики
f1 = f1_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")

# Сохранение метрик
report = classification_report(true_labels, predictions, target_names=['норма', 'лёгкое', 'серьёзное'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_excel('classification_report.xlsx', index=True)
print("Отчёт о метриках сохранён в 'classification_report.xlsx'")

# Визуализация потерь
plt.plot(range(1, len(loss_values) + 1), loss_values, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Сохранение модели
model_save_path = "improved_trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Модель сохранена в {model_save_path}")

