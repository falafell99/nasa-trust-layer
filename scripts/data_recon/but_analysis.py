import csv, sys, glob, os
base = sys.argv[1]
GROUPS = {1: 0, 2: 3, 3: 6, 4: 9}  # annotator -> first column (0-based), per ann_reader.m sl=[1 4 7 10]

def read_ann(path):
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            rows.append(r)
    return rows

def groups(rows):
    out = {}
    for a, c in GROUPS.items():
        segs = []
        for r in rows:
            if len(r) > c + 2 and r[c].strip() != '' and r[c+1].strip() != '' and r[c+2].strip() != '':
                segs.append((int(float(r[c])), int(float(r[c+1])), int(float(r[c+2]))))
        out[a] = segs
    return out

print("id  nrows  ECG_n  | per annotator: n_segs, last_end | consensus(a4): labeled_frac, q1%,q2%,q3% of total ; q3_samples ; contiguous/overlap check")
res = {}
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d, d + "_ANN.csv")
    if not os.path.isfile(p):
        continue
    rows = read_ann(p)
    g = groups(rows)
    hea = open(os.path.join(base, d, d + "_ECG.hea")).readline().split()
    n_ecg = int(hea[3])
    ncols = {len(r) for r in rows}
    cons = g[4]
    cnt = {0: 0, 1: 0, 2: 0, 3: 0}
    other = 0
    overlap = 0
    prev_end = 0
    gaps = 0
    for s, e, c in cons:
        if s != prev_end + 1:
            gaps += 1
        prev_end = e
        if c in cnt:
            cnt[c] += e - s + 1
        else:
            other += e - s + 1
    # also per annotator labeled fraction
    per = {}
    for a in (1, 2, 3, 4):
        lab = sum(e - s + 1 for s, e, c in g[a] if c != 0)
        per[a] = (len(g[a]), g[a][-1][1] if g[a] else None, lab / n_ecg)
    lab = n_ecg and (cnt[1] + cnt[2] + cnt[3]) / n_ecg
    res[d] = dict(n=n_ecg, cnt=cnt, lab=lab, cons_last=cons[-1][1], ncols=ncols, gaps=gaps, per=per, other=other)
    print(d, len(rows), n_ecg, "|", {a: (v[0], v[1]) for a, v in per.items()}, "| lab=%.4f q0=%.4f q1=%.4f q2=%.4f q3=%.4f q3_s=%d cons_last=%s gaps=%d ncols=%s other=%d" % (
        lab, cnt[0] / n_ecg, cnt[1] / n_ecg, cnt[2] / n_ecg, cnt[3] / n_ecg, cnt[3], cons[-1][1], gaps, sorted(ncols), other))

tot = sum(r['n'] for r in res.values())
print("\nTOTAL ECG samples", tot, "= %.2f h" % (tot / 1000 / 3600))
lab_all = sum(sum(r['cnt'][c] for c in (1, 2, 3)) for r in res.values())
print("Consensus labeled (class!=0) / total samples: %.4f" % (lab_all / tot))
for c in (1, 2, 3):
    s = sum(r['cnt'][c] for r in res.values())
    print("class", c, "share of total %.4f, share of labeled %.4f" % (s / tot, s / lab_all))
print("\nRanking by consensus labeled fraction:")
for d, r in sorted(res.items(), key=lambda kv: -kv[1]['lab']):
    print(d, "labeled=%.4f hours_labeled=%.2f q3_samples=%d (%.1f s)" % (r['lab'], r['lab'] * r['n'] / 1000 / 3600, r['cnt'][3], r['cnt'][3] / 1000))
