# Бот для генерации анекдотов

В папке experiments можно ознакомиться с ноутбуком для парсинга данных `load_data.ipynb` и с ноутбуками обучения моделей:
1. пока доступна только llm на основе статистик:
- `stat-llm.ipynb`: первая версия на необработанных данных
- `stat-llm-v2.ipynb`: новая версия на сильно улучшенных данных (предобработка производилась в `load_data.ipynb`)

Модель обучалась на двух датасетах:
- [Russian Jokes](https://www.kaggle.com/datasets/konstantinalbul/russian-jokes)
- [Собственный](https://www.kaggle.com/datasets/boogiewoogieqq/vk-anekdots)

Чтобы запустить код предварительно необходимо скачать [модель](https://drive.google.com/file/d/15uao0yIp5wUraUsEvbOe65Y_PqWrA-zj/view?usp=sharing) и добавить файл в папку `models/stat_lm`.
