import os
import csv

def add_to_count(root, filename, count_dict, weight):
    with open(os.path.join(root, filename), mode='r') as file:
        csvFile = csv.reader(file)
        for idx, lines in enumerate(csvFile):
            if idx == 0:
                continue
            count_dict[f'{idx - 1}'][int(lines[1])] += weight

# count dictionary for all images with shape (keys: 0 ~ 8099, values: [0,0,0,0,0,0])
count = {}
for i in range(8100):
    count[f'{i}'] = [0, 0, 0, 0, 0, 0]

# ensemble the no_pt results of all models
for filename in os.listdir('submission/etc'):
    if filename.startswith('ViT-bigG-14-CLIPA_datacomp1b'):
        add_to_count('submission/etc', filename, count, 2.8)
    elif filename.endswith('.csv'):
        add_to_count('submission/etc', filename, count, 1)

for filename in os.listdir('submission/no_pt'):
    if filename.startswith('ViT-bigG-14-CLIPA_datacomp1b'):
        add_to_count('submission/no_pt', filename, count, 2.8)
    elif filename.endswith('.csv'):
        add_to_count('submission/no_pt', filename, count, 1)

for filename in os.listdir('submission/pt'):
    if filename.startswith('ViT-bigG-14-CLIPA_datacomp1b'):
        add_to_count('submission/pt', filename, count, 2.8)
    elif filename.endswith('.csv'):
        add_to_count('submission/pt', filename, count, 1)

# for filename in os.listdir('clip_adapter_results'):
#     if filename.startswith('ViT-bigG-14-CLIPA_datacomp1b'):
#         add_to_count('clip_adapter_results', filename, count, 1)
#     elif filename.endswith('.csv'):
#         add_to_count('clip_adapter_results', filename, count, 1)

# write ensemble data to the csv file
with open('exp321.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id_idx', 'label'])
    for line in range(8100):
        print(count[f'{line}'])
        ls = count[f'{line}']
        max_idx = ls.index(max(ls))
        count[f'{line}'] = max_idx
        writer.writerow([line, count[f'{line}']])