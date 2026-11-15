#!/bin/sh
# Скачивает VTaC v1.1 annotated_waveforms (3.6 GB) с physionet.org; пропускает уже скачанное. Запуск из data/vtac.
BASE=https://physionet.org/files/vtac/1.1
dl() { f="$1"; [ -s "$f" ] || { mkdir -p "$(dirname "$f")"; curl -sf --retry 3 -o "$f" "$BASE/$f"; }; }
export BASE; export -f dl 2>/dev/null
sed -e 's/$/.hea/' RECORDS > /tmp/vtac_list.txt; sed -e 's/$/.dat/' RECORDS >> /tmp/vtac_list.txt
cat /tmp/vtac_list.txt | xargs -P 32 -I{} sh -c 'f={}; [ -s "$f" ] || { mkdir -p "$(dirname "$f")"; curl -sf --retry 3 -o "$f" "'$BASE'/$f"; }'
echo done
