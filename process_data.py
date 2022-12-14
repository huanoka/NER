import json
import os
import re


def gen_label_sequence(path, split):
    output_file = open(path + '/' + 'labeling/' + split + '_split', 'w', encoding='utf-8')
    output_file.write('text,labels\n')
    for file_name in os.listdir(path + '/' + split):
        full_name = path + '/' + split + '/' + file_name
        with open(full_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                json_obj = json.loads(line)
                text = json_obj['text']
                text = re.subn(r',', '，', text)[0]
                labels = ['O' for _ in range(len(text))]
                for label in json_obj['labels']:
                    start = label[0]
                    end = label[1]
                    attr = label[2]
                    if attr == 'materiel':
                        attr = 'material'
                    if end - start == 1:
                        labels[start] = 'S-' + str(attr)
                        continue
                    labels[start] = 'B-' + str(attr)
                    labels[end-1] = 'E-' + str(attr)
                    for i in range(start+1, end-1):
                        labels[i] = 'I-' + str(attr)
                out_str = str(text) + ','
                for ll in labels:
                    out_str += str(ll) + ' '
                out_str = out_str[:-1]
                out_str += '\n'
                output_file.write(out_str)


# gen_label_sequence("data", 'dev')
gen_label_sequence("data", 'test')
# gen_label_sequence("data", 'train')
