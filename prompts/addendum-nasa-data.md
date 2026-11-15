# Дополнение: данные NASA и партнёров на странице челленджа

Вставь после общего контекста проекта (`context-pack-for-llms.md`).

```
ROLE: skeptical research advisor. Search the web, read primary sources. Label every claim [Established] / [Likely] / [Guess] and give a link. If you cannot verify something, say so.

TASK: Open the NASA International Space Apps Challenge 2026 page for the challenge "Create Health Monitoring Software for Astronauts on Space Missions" (spaceappschallenge.org). Copy EXACTLY, with links:
1. the list of datasets, resources and "Suggested data" from NASA and partner agencies (CSA, ESA, JAXA, others) shown on the challenge page;
2. for each item: what it contains, size, file format, license, whether it can be downloaded without registration, whether it has time series (ECG, PPG, heart rate, blood pressure, accelerometer) and labels;
3. any rule on the page or in the 2026 Judging & Awards Guide / Project Submission Guide about whether using NASA or partner-agency data is REQUIRED for global judging, and whether it applies to local events;
4. which of the listed items could realistically be used by a 3-person team in 8 hours for an offline signal-quality / alarm-verification prototype, and which would only be context (not validation).

Do not summarise from secondary blogs. If the page cannot be opened, say so and give the closest primary source you did open. Do not invent dataset names.
```

Ответ положить в `docs/raw/` и обновить Q1 в `docs/09`.
