files = ['/home/jcoves/elit/elit/dat/ner/eng.trn.bilou','/home/jcoves/elit/elit/dat/ner/eng.dev.bilou','/home/jcoves/elit/elit/dat/ner/eng.tst.bilou']
chars = set()
for file in files:
    f = open(file, 'r')
    for i, line in enumerate(f):
        if i>0 and len(line.split('\t')) > 3:
            chars.update(list(line.split('\t')[0]))
    f.close()
print(len(chars))
for i, label in enumerate(chars):
    print("'%s':%d, " % (label, i+1), end='')
