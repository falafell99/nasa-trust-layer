import sys, os, glob, collections, csv
d = sys.argv[1]  # dir containing *.hea and ALARMS
alarms = {}
raw = open(os.path.join(d, 'ALARMS')).read().splitlines()
print('ALARMS first 5 lines:', raw[:5], '| total lines', len(raw))
header = raw[0].split(',')
for line in raw:
    p = [x.strip() for x in line.split(',')]
    if len(p) >= 3 and p[0] not in ('Record', 'record'):
        alarms[p[0]] = (p[1], p[2])
print('label raw values:', collections.Counter(v[1] for v in alarms.values()))
print('alarm types raw:', collections.Counter(v[0] for v in alarms.values()))
heas = sorted(glob.glob(os.path.join(d, '*.hea')))
print('hea files', len(heas), 'alarms', len(alarms))
info = {}
for h in heas:
    ls = open(h).read().splitlines()
    first = ls[0].split()
    rec = first[0]
    nsig, fs, ns = int(first[1]), float(first[2]), int(first[3])
    names = []
    for l in ls[1:1 + nsig]:
        names.append(l.split()[-1])
    info[rec] = (nsig, fs, ns, names, [l for l in ls[1 + nsig:] if l.startswith('#')])
print('fs values', collections.Counter(v[1] for v in info.values()))
print('nsamples values', collections.Counter(v[2] for v in info.values()))
print('nsig values', collections.Counter(v[0] for v in info.values()))
print('sig name sets', collections.Counter(tuple(v[3]) for v in info.values()))
print('records w/o ALARMS entry', len(set(info) - set(alarms)), '; in ALARMS w/o hea', len(set(alarms) - set(info)))
ex = next(iter(info))
print('example comments', ex, info[ex][4])
TRUE = {'1', 'true', 'True', 'TRUE'}
def istrue(x): return x in TRUE
# PLETH / ABP / RESP presence
def has(r, n): return n in info[r][3]
recs = sorted(info)
c = collections.Counter()
for r in recs:
    c[('PLETH' if has(r, 'PLETH') else '-', 'ABP' if has(r, 'ABP') else '-', 'RESP' if has(r, 'RESP') else '-')] += 1
print('presence (PLETH,ABP,RESP):', c)
n_pleth = sum(has(r, 'PLETH') for r in recs)
n_abp_only = sum((not has(r, 'PLETH')) and has(r, 'ABP') for r in recs)
n_neither = sum((not has(r, 'PLETH')) and (not has(r, 'ABP')) for r in recs)
n_both = sum(has(r, 'PLETH') and has(r, 'ABP') for r in recs)
print('PLETH present %d / %d; PLETH absent %d; ABP-only (no PLETH) %d; neither %d; both %d' % (n_pleth, len(recs), len(recs) - n_pleth, n_abp_only, n_neither, n_both))
ecg_names = lambda r: [n for n in info[r][3] if n not in ('PLETH', 'ABP', 'RESP')]
print('ECG channel counts', collections.Counter(len(ecg_names(r)) for r in recs))
print('ECG name tuples', collections.Counter(tuple(ecg_names(r)) for r in recs))
# true/false by type
tab = collections.defaultdict(lambda: [0, 0])
for r in recs:
    t, l = alarms[r]
    tab[t][0 if istrue(l) else 1] += 1
print('\nType | True | False | total')
T = F = 0
for t, (a, b) in sorted(tab.items()):
    print(t, a, b, a + b); T += a; F += b
print('ALL', T, F, T + F, ' true share %.4f' % (T / (T + F)))
# PLETH subset by label/type
print('\nAmong records WITH PLETH:')
tab2 = collections.defaultdict(lambda: [0, 0])
for r in recs:
    if has(r, 'PLETH'):
        t, l = alarms[r]
        tab2[t][0 if istrue(l) else 1] += 1
T2 = F2 = 0
for t, (a, b) in sorted(tab2.items()):
    print(t, a, b, a + b); T2 += a; F2 += b
print('ALL', T2, F2, T2 + F2)
print('\nAmong records WITHOUT PLETH:')
tab3 = collections.defaultdict(lambda: [0, 0])
for r in recs:
    if not has(r, 'PLETH'):
        t, l = alarms[r]
        tab3[t][0 if istrue(l) else 1] += 1
for t, (a, b) in sorted(tab3.items()):
    print(t, a, b, a + b)
# record prefix by type
print('\nRecord id prefix by type:', collections.Counter((r[0], alarms[r][0]) for r in recs))
