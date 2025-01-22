from transformers import BertForSequenceClassification
import torch
from transformers import BertTokenizer
import pandas as pd

# Загрузка модели
model_path = "improved_trained_model.pth"  # Укажите путь к сохраненной модели
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.load_state_dict(torch.load(model_path))
model.eval()  # Перевод в режим инференса
print("Модель успешно загружена.")
# Загрузка новых данных
new_data_file = "очищенный_файл.xlsx"
new_data = pd.read_excel(new_data_file)

# Очистка текста
new_data['ParamValue_cleaned'] = new_data['ParamValue'].str.replace(r'&nbsp;', ' ').str.lower().str.strip()

# Токенизация
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
new_data['tokens'] = new_data['ParamValue_cleaned'].apply(
    lambda x: tokenizer.encode_plus(x,
                                    add_special_tokens=True,
                                    max_length=50,
                                    truncation=True,
                                    padding='max_length',
                                    return_attention_mask=True)
)

# Создание DataLoader
input_ids = torch.tensor([item['input_ids'] for item in new_data['tokens']], dtype=torch.long)
attention_masks = torch.tensor([item['attention_mask'] for item in new_data['tokens']], dtype=torch.long)

dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
predictions = []

for batch in dataloader:
    input_ids, attention_masks = batch
    with torch.no_grad():  # Отключаем градиенты для предсказаний
        outputs = model(input_ids, attention_mask=attention_masks)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1)  # Находим класс с максимальной вероятностью
    predictions.extend(predicted_classes.cpu().numpy())  # Переносим результаты на CPU

# Соответствие числовых меток и категорий
label_map = {0: 'норма', 1: 'лёгкое', 2: 'серьёзное'}
new_data['Predicted_Label'] = [label_map[pred] for pred in predictions]

# Сохранение в файл
output_file = "predictions_new_data.xlsx"
new_data.to_excel(output_file, index=False)
print(f"Предсказания сохранены в {output_file}")
