# Neural Machine Translation with LSTM and Attention

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=weightsandbiases)](https://wandb.ai)

Нейронная сеть для машинного перевода (немецкий → английский) на основе **bidirection LSTM** с **механизмом внимания**. Поддерживает регулируемый teacher forcing и beam search.

## Особенности
- **Архитектура**: Двунаправленный LSTM-encoder + LSTM-decoder с attention.
- **Функционал**:
  - Динамический teacher forcing.
  - Beam search для инференса.
  - Оценка качества через BLEU.
  - Интеграция с Weights & Biases.
- **Обработка данных**:
  - Автоматическая фильтрация редких слов.
  - Замена чисел на `<NUM>`
  - Замена межязыковых токенов на `<SUB>`.

## Отчет о проделанной работе
1. [Работа с данными](report/dataset.md)
2. [Архитекрутра модели](report/model.md)
3. [Train loop](report/train.md)
4. [Beam search](report/beam-search.md)
5. [Эксперименты с гиперпараметрами](report/experiments.md)
6. [BLUE](report/bleu.md)
7. [Пути улучшения](report/plan.md)
