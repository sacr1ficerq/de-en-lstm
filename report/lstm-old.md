## LSTM

Архитектура, которую я обучил к чекпоинту лежит в файле `lstm.py`.

У меня не сохранились графики обучения и ошибки до этой модели, но они не сильно интересны, так как только на ней я пробил 20 BLEU4.

<pre class="vditor-reset" contenteditable="true" spellcheck="false">
    <img src="images/lstm-sorted-batches-losses.png"/>
</pre>

У модели следующая архитектура:

```python
  (src_embedding): Embedding(55315, 64)
  (trg_embedding): Embedding(34047, 64)
  (encoder): LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (encoder_output_proj): Linear(in_features=256, out_features=128, bias=True)
  (decoder): LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.1)
  (encoder_hidden_proj): (
    (0-1): 2 x Linear(in_features=256, out_features=128, bias=True)
  )
  (encoder_cell_proj): (
    (0-1): 2 x Linear(in_features=256, out_features=128, bias=True)
  )
  (fc): Linear(in_features=256, out_features=34047, bias=True)
```

Внутри форварда я проецирую долговременную и краткосрочную память двунаправленного енкодера (`hidden`, `cell`) в пространство однонаправленного декодера и применяю некоторое подобие внимания:

```python-repl
# attention
energy = torch.bmm(decoder_outputs, encoder_outputs.transpose(1,2))  
attention = F.softmax(energy,dim=-1)
context = torch.bmm(attention, encoder_outputs)
combined = torch.cat([decoder_outputs, context],dim=2)
logits = self.fc(combined)
```

в `energy` лежит тензор скалярных проихведений векторов декодера, который мы нормируем и с этими весами берем спроецированный выход енкодера как контекст. Дальше проецируем объединение этих тензоров в логиты.
