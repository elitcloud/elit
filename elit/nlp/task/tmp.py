f=open('/home/jcoves/elit/elit/dat/ner/eng.trn.bilou')
labels = set()
for line in f:
    if len(line.split('\t')) > 3:
        labels.add(line.split('\t')[2])
print(len(labels))
for i,label in enumerate(labels):
    print("'%s':%d, " % (label, i), end='')
f.close()