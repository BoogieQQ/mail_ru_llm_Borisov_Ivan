# Бот для генерации анекдотов

В папке experiments можно ознакомиться с ноутбуком для парсинга данных `load_data.ipynb` и с ноутбуками обучения моделей:
1. пока доступна только llm на основе статистик:
- `stat-llm.ipynb`: первая версия на необработанных данных
- `stat-llm-v2.ipynb`: новая версия на сильно улучшенных данных (предобработка производилась в `load_data.ipynb`)

Модель обучалась на двух датасетах:
- [Russian Jokes](https://www.kaggle.com/datasets/konstantinalbul/russian-jokes)
- [Собственный](https://www.kaggle.com/datasets/boogiewoogieqq/vk-anekdots)

Чтобы запустить код предварительно необходимо скачать [модель и токенайзер](https://drive.google.com/drive/folders/1nHu3oWL4WTf3iLuokiNIllQzRAaxNEz1?usp=sharing) и добавить файлы в папку `models/stat_lm`.
