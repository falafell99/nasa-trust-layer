import csv, os, sys
base=sys.argv[1]
G={1:0,2:3,3:6,4:9}
def segs(rows,c):
    return [(int(r[c]),int(r[c+1]),int(r[c+2])) for r in rows if len(r)>c+2 and r[c+2].strip()!='']
def label_at(sg, points):
    # sg sorted, points sorted; return list of class per point (0 if none)
    out=[];i=0
    for p in points:
        while i<len(sg) and sg[i][1]<p: i+=1
        out.append(sg[i][2] if i<len(sg) and sg[i][0]<=p<=sg[i][1] else None)
    return out
tot={1:0,2:0,3:0}; N=0
for d in ['100001','105001','111001']:
    rows=list(csv.reader(open(f'{base}/{d}/{d}_ANN.csv')))
    S={a:segs(rows,c) for a,c in G.items()}
    for a in S: S[a].sort()
    # sample boundaries
    bps=sorted({s for a in S for s,e,c in S[a]}|{e+1 for a in S for s,e,c in S[a]})
    # evaluate on each elementary interval [bps[k], bps[k+1]-1]
    pts=bps[:-1]
    L={a:label_at(S[a],pts) for a in S}
    agree=maj=eq_any=0; dur_tot=0; part={1:0,2:0,3:0}
    for k,p in enumerate(pts):
        w=bps[k+1]-p
        v=[L[a][k] for a in (1,2,3)]; c4=L[4][k]
        if c4 is None: continue
        dur_tot+=w
        part[c4]+=w
        # majority of annotators 1-3
        from collections import Counter
        cnt=Counter(x for x in v if x is not None)
        top,n=(cnt.most_common(1)[0] if cnt else (None,0))
        if n>=2:
            maj+=w
            if top==c4: agree+=w
    print(d,'duration labeled',dur_tot,'majority(>=2 of a1-3) exists on %.4f'%(maj/dur_tot),'a4==majority on %.4f of those'%(agree/maj))
    for c in (1,2,3): tot[c]+=part[c]
    N+=dur_tot
print('pooled 3 records: total samples',N,'= %.2f h'%(N/1000/3600))
for c in (1,2,3): print('class',c,'%.4f'%(tot[c]/N), 'hours %.2f'%(tot[c]/1000/3600))
