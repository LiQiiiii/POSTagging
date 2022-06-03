from Vocab import POSVocab, CharVocab

words_set = set()
char_set = set()
pos_set = set()
#create_vocabs
with open('dev.tsv', 'r', encoding='utf-8', errors='ignore') as fin:
    for line in fin:
        if(line == "\n"):
            continue
        if(len(line.strip().split('\t')) != 2):
            continue
        wd, pos = line.strip().split('\t')
        char_set.update([ch.strip() for ch in wd])
        words_set.add(wd)
        pos_set.add(pos)


# print(words_set)
print(len(char_set))
# print(pos_set)
# print(len(pos_set))

# file = open("pos_set.txt", 'a', encoding='utf-8')  # 写入模式
# for n in pos_set:
#     file.write(n + " ")
# file.close()
