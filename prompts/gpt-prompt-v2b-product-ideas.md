# Промпт для GPT, часть 2: видение продукта и техническая реализация

Отправь в тот же чат, где GPT уже получил `gpt-prompt-v2-full.md` (чтобы контекст был у него). Если чат новый, сначала загрузи тот файл.

```
ROLE: senior engineer and product thinker for safety-critical alerting on wearables. Label every claim [Established] / [Likely] / [Guess]. Do not agree by default. Start with the weakest part of our current product concept.

Using the full context I gave you, forget our current plan for a moment and answer as if you were the founder who must ship a working offline prototype in about 12 hours with 3 people and AI coding agents, and win a local jury of space engineers and a few biomedical mentors.

1. PRODUCT VISION. In 5 sentences, how do you see the product? Who is the user (astronaut, crew medical officer, ground team), what exact moment of use, what do they see and do in 10 seconds? What would make a jury say "this is real" rather than "another ECG classifier"?
2. THREE ALTERNATIVE PRODUCT CONCEPTS, each different in kind (not variations of our Trust Layer), each buildable in 12 hours on the datasets in docs/04 only. For each: one-line pitch, what it measures, which dataset, what the demo screen shows, the main way it can fail, and what you cannot claim.
3. OUR CONCEPT, BEST VERSION. How would you make the Trust Layer stronger without adding scope? Name the 3 changes with the highest value per hour.
4. TECHNICAL ARCHITECTURE. Propose a concrete stack for a laptop-only offline prototype: language, libraries, streaming design (how a recorded signal becomes a live feed), windowing, feature set (give 15-25 concrete features for ECG, PPG and quality), model type, how thresholds are chosen, how latency and memory are measured, how the UI gets results (websocket / local web app / terminal), and how everything is packaged to run on a second laptop in one command. Mention what you would NOT use.
5. UI. Sketch the main screen in text (layout, what is shown, what is hidden), including how uncertainty and the original alarm are shown. State which single element carries most of the "trust" message.
6. THE DEMO THAT WINS. Design a 90-second live demo using only our real data (two separate scenes are allowed). What exactly happens second by second, what is the "aha" moment, what is the backup if live fails?
7. WHAT IS TECHNICALLY IMPRESSIVE BUT CHEAP. List up to 5 things that look sophisticated to a judge but cost under 1 hour each, and 5 things that look sophisticated but secretly cost more than 4 hours.
8. WHAT YOU WOULD ASK THE JURY OR A MENTOR in a 3-minute chat to learn whether the product direction is right.

Constraints: no new datasets unless you opened the page and the files are under 1 GB, no registration. No hype words, no filler. Be concrete and numeric where you can. If you disagree with our framing, say so first.
```

После ответа: положи его в `docs/raw/` (следующий номер), добавь строку в `docs/raw/INDEX.md` и в `docs/07`, и пришли мне. Идеи из пунктов 2, 3 и 6 оформляются как предложения в `docs/10`, а не как решения.
