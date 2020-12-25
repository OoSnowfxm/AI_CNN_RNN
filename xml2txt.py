import xml.etree.ElementTree as ET
from string import punctuation
from nltk.corpus import stopwords

tree = ET.parse('data/Laptops_trial_data.xml')
root = tree.getroot()
sentences = []

for sentence in root.iter("sentence"):
    text = sentence.find("text")
    try:
        aspectTerms=sentence.find('aspectTerms')
        aspects = []
        for aspectTerm in aspectTerms.findall('aspectTerm'):
            term = aspectTerm.get('term')
            aspects.append(aspectTerm.attrib)
        sentences.append({"text":text.text, "aspects":aspects})
    except AttributeError:
        sentences.append({"text":text.text, "aspects": None})

print(sentences[0])
out = open("data/trial1.txt","w", encoding="utf-8")

data = []
pad = 0
global_aspect_count = 0
for sentence in sentences:
    tupledata = []
    aspects = sentence["aspects"]
    text = sentence["text"]
    tupledata.append(text)
    if aspects is None:
        pad+=1
        text = text.strip()
        words = text.split(" ")
        for word in words:
            if word.strip() != "":
                out.write(word+"\t"+"O"+"\n")
        out.write("\n")
    else:
        pad+=1
        dict = {}
        for aspect in aspects:
            target = aspect["term"]
            from_ = int(aspect["from"])
            to_ = int(aspect["to"])
            if target != "NULL" and from_ not in dict.keys():
                dict[from_] = [target,from_,to_]
            elif from_ in dict.keys():
                print(text)
                print(target == dict[from_][0])


        keys = sorted(dict)

        #print(dict)
        if len(keys) > 0:
            dump = ""
            last_end = 0
            counter = 0

            for key in keys:
                    global_aspect_count += 1
                    vals = dict[key]

                    target = vals[0]
                    from_ = vals[1]
                    to_ = vals[2]

                    aspect_ = text[from_:to_]
                    temp = text[last_end:from_]
                    last_end = to_

                    if aspect_ == target:
                        storage = ""
                        aspect = target.split(" ")
                        i = 0
                        for asp in aspect:
                            if i == 0:
                                storage = storage + asp + "\t" + "B" + "\n"
                                i+=1
                            else:
                                storage = storage + asp + "\t" + "I" + "\n"
                                i+=1
                        temp+=storage
                        dump+=temp
                        if counter == len(keys) -1:
                            dump+=text[to_:]
                        counter+=1
                    else:
                        counter+=1

            if dump!= "":
                dump = dump.replace(" ","\t"+"O"+"\n")
                dump+= "\t"+"O"
                out.write(dump+"\n\n")
        else:
            text = text.strip()
            words = text.split(" ")
            for word in words:
                if word.strip() != "":
                    out.write(word + "\t" + "O" + "\n")
            out.write("\n")

out.close()

f = open("data/trial1.txt","r", encoding="utf-8")
out = open("data/trial.txt","w", encoding="utf-8")

stop_words = set(stopwords.words('english'))

for line in f:
    if line.strip() != "":
        #line = line.replace("..."," ")
        line1 = line.split("\t")
        line2 = ''.join(c for c in line1[0] if (c not in punctuation))
        if line2.strip() == "" or line2 in stop_words:
            continue
        else:
            out.write(line2+"\t"+line1[1])
    else:
       out.write("\n")
out.close()
f.close()

