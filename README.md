# NASA Space Apps 2026, Budapest: Trust Layer (команда «Zakaz Konina»)

**Демо (3 строки):**
1. Установка: `python3 -m venv .venv && .venv/bin/pip install -r requirements-demo.txt`.
2. Запуск: `./run_demo.sh` — после установки зависимостей сеть не нужна; демо использует только `demo_data/` и модели из `models/`.
3. Открой http://127.0.0.1:8000: главный экран показывает wearable-сценарий. Исходные ECG-кейсы B1/B2/B3 и BUT QDB A доступны по http://127.0.0.1:8000/cases.

В репозитории лежит только нужный небольшой набор сцен (примерно 2,75 MB), не полные VTaC и BUT QDB. Атрибуция, состав и лицензии: [`demo_data/ATTRIBUTION.md`](demo_data/ATTRIBUTION.md). Модель V5 и параметры V5/Q-gate находятся в git: `models/v5_lgbm.txt`, `models/v5_config.json`, `models/quality_v2.json`.

## Apple Watch / wearable-демо

- **Demo scenario** — явно подписанные синтетические показания пульса, дыхания, SpO₂ и температуры запястья. Кашель добавляется только вручную как сообщение человека.
- **Свои данные** — iPhone → Health → профиль → **Export All Health Data**; распакуй ZIP и выбери `export.xml` на главном экране. Импортируются только записи с источником Apple Watch: HeartRate, RespiratoryRate, OxygenSaturation, AppleSleepingWristTemperature. Сырые данные не сохраняются; последние 240 показаний остаются в памяти процесса до перезапуска. Импорт принимает XML до 200 MB.
- **Передача из iPhone Shortcuts по локальной сети:** задай секрет и LAN-host при запуске: `WATCH_DEMO_TOKEN=<секрет> WATCH_DEMO_HOST=0.0.0.0 ./run_demo.sh`. В Shortcuts используй **Find Health Samples** → извлеки значение и время → **Get Contents of URL** (`POST`, JSON) по адресу `http://<IP-ноутбука>:8000/api/watch/samples`, заголовок `X-Watch-Demo-Token: <секрет>`. Тело запроса:

```json
{"samples":[{"metric":"heart_rate","value":78,"time":"2026-11-15T14:00:00+01:00"}]}
```

Поддерживаются `heart_rate` (bpm), `respiratory_rate` (breaths/min), `oxygen_saturation` (%) и `wrist_temperature` (°C). `GET /api/watch/latest` возвращает последние показания; страница обновляется каждые 5 секунд. Ноутбук и iPhone должны находиться в одной сети; без `WATCH_DEMO_TOKEN` приём POST выключен. Для живой передачи нужна настройка Shortcut на iPhone, приложение watchOS в репозитории отсутствует.

Показатели Apple Watch отображаются отдельно. V5 обучена на ECG+PPG тревогах и не вызывается на Health-показаниях. Экран описывает изменения и отсутствие данных, не ставит диагноз. Apple Watch ECG записывается отдельным действием владельца; обычный экспорт показателей не создаёт непрерывную ECG-кривую для V5.

## Detector attempt 2: frozen exploratory protocol

- [Точно] Scope: record-level nested cross-validation only; records used as hold-out in the first detector run are excluded before signal loading. Every input sampling rate is checked against the WFDB header, then signals are anti-aliased and resampled to 250 Hz. MIT-BIH annotations are restricted to VT.
- [Точно] Window task: 8 s windows with 4 s stride. Positive means VT, VF or VFIB annotation overlaps at least 4 s; boundary windows with shorter nonzero overlap are ignored. Ventricular flutter is excluded from the positive class.
- [Точно] Features: RR rate/regularity, spectral concentration, threshold-crossing interval proxy (positive crossings of 10% peak amplitude), VF-filter leakage (half-period cancellation residual), sample entropy (m=2, r=0.2 SD, computed after 2x downsampling), and median QRS-width proxy. Logistic regression, CPU.
- [Точно] Outer evaluation uses 5 folds of whole records; an inner 4-fold CV on outer-training records selects one threshold per outer fold by mean record-level Youden J. Threshold selection never uses the evaluated outer fold.
- [Точно] Episode rule: consecutive predicted-positive windows stay in one alarm until 30 consecutive seconds without a positive window; the alarm timestamp is the end of the first positive 8 s window. Episode matching requires alarm time inside an annotated VT/VF interval. False alarms count only detections outside annotated episodes; duplicate detections within an event are ignored, and event time is excluded from background hours.
- [Точно] Contiguous target annotations are merged across VT/VF class changes into one dangerous-rhythm episode; an intervening non-target gap starts a new episode.
- [Точно] Result label: **exploratory, cross-validated, not held-out**. Window metrics are macro-averaged by record; uncertainty is bootstrap by record.

Feature names follow common VT/VF signal-processing families; VF leakage is the residual after half-period cancellation, and sample entropy/thresholded binary complexity are established candidate features. This is not a reproduction of any published detector: [Zhang et al., 1999](https://pubmed.ncbi.nlm.nih.gov/10230133/); [Teixeira de Barros Moraes et al., 2003](https://doi.org/10.1109/CIC.2002.1166745).

Общая память проекта для людей и их агентов. Снимок состояния: **2026-11-15, около 03:00 по Будапешту**. Подача до **16:00**, питчи в **17:00** (3 минуты).

Челлендж: «Create Health Monitoring Software for Astronauts on Space Missions». Фокус: Луна/Марс (не МКС). Цель: локальный приз, настоящий работающий прототип, а не красивый дашборд.

## Где мы сейчас (коротко)

- **Идея (одобрена лидом, вариант B-lite):** офлайн-слой «Trust Layer» между датчиком и экипажем. Простое правило поднимает тревогу по ЭКГ, наш слой отвечает: (1) можно ли верить сигналу (шлюз качества), (2) настоящее ли событие (проверка по двум отведениям ЭКГ и PPG). Исходная тревога никогда не скрывается.
- **Дизайн-документ:** `docs/03-product-design.md`. Лид сказал «ок, фиксируем» про вариант B, но **слово «одобряю» про сам документ ещё не сказано**. Пока этого нет: код продукта не пишем. Скачивание данных, чтение файлов, подсчёт меток, проверки доступности делать можно и нужно.
- **Данные:** план в `docs/04-data-plan.md`. Три основных набора: CinC 2015 (тревоги), BUT QDB (качество ЭКГ + акселерометр), MIT-BIH NST (контролируемый шум). Запасной второй набор тревог: VTaC v1.1. Никто из нас и никто из внешних моделей **не скачал бинарные файлы**, поэтому «качается за 2 минуты» не проверено. Первые 30 минут команды: go/no-go по данным (раздел 5 в `docs/04-data-plan.md`).
- **Главный риск:** шлюз качества учится на BUT QDB (носимый датчик, здоровые, нет тревог), а применяется к CinC 2015 (реанимация, нет меток качества и движения). Это перенос между доменами, выигрыша может не быть. Честный негативный результат допустим.
- **Чего у нас нет и не будет:** открытых сырых ЭКГ астронавтов. Всё, что мы показываем, это земные прокси. Так и говорим.

## Порядок чтения для нового человека или агента

1. `AGENTS.md`: правила работы (метки уверенности, честность чисел, как писать прогресс).
2. `docs/01-situation-and-goals.md`: ситуация, сроки, критерии жюри.
3. `docs/03-product-design.md`: что строим (контракт интерфейсов в разделе 6).
4. `docs/04-data-plan.md`: какие данные, что скачать, ловушки.
5. `docs/05-evaluation-protocol.md`: как считаем результат и что можно заявлять.
6. `docs/11-roles-and-timeline.md`: кто что делает и когда.
7. По необходимости: `docs/02-problem-evidence.md` (числа о проблеме), `docs/06-claims-and-prior-art.md` (что можно говорить жюри), `docs/07-external-model-feedback.md` (что сказали GPT/Perplexity/Gemini и где они расходятся), `docs/08-pitch-and-demo.md`, `docs/09-open-questions-and-risks.md`, `docs/10-decision-log.md`, `docs/12-sources.md`, `docs/13-glossary.md`.

## Структура репозитория

```
README.md                      этот файл (состояние и навигация)
AGENTS.md                      правила для агентов и людей
docs/                          канонические документы (меняются только с записью в decision-log)
docs/raw/                      сырые тексты внешних моделей и документов без правок (источник правды для деталей)
prompts/                       промпты для внешних моделей и контекст-пакет
progress/STATUS.md             чек-лист этапов (правит лид)
progress/logs/<имя>.md         личный журнал каждого человека/агента (только дописывать)
progress/handoffs/TEMPLATE.md  шаблон передачи работы между людьми
src/                           код (пока пусто: ждём одобрения дизайна)
data/                          данные (в git не кладём, см. .gitignore)
```

## Как делиться прогрессом между людьми и агентами

- Каждый пишет **только в свой** `progress/logs/<имя>.md` (дописывать вниз, не править старое). Так не бывает конфликтов слияния.
- Перед началом работы: `git pull --rebase`. После: небольшой коммит и `git push`.
- Любое решение, меняющее дизайн, данные или метрики, добавляется в `docs/10-decision-log.md` отдельным коммитом.
- Числа результатов: только с пометкой «наше измерение», датой, версией кода и командой, которой получены (см. `AGENTS.md`).

## Что показывать жюри и как не навредить себе

Читать `docs/06-claims-and-prior-art.md` до написания слайдов. Коротко: не говорим «первые в мире», не называем результат валидацией для космоса, не смешиваем знаменатели (ложные среди тревог, артефакты среди записей, забракованные окна).
