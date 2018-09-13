import csv
from os.path import join
from os import getcwd

data_dir = join(getcwd(), 'MURA-v1.1')
train_csv = join(data_dir, 'valid_image_paths.csv')
File_root = open(train_csv)
File_reader = csv.reader(File_root)

# 建立空字典
result = []
for i, item in enumerate(File_reader):
    label = 1 if 'positive' in str(item).split('_')[-1].split('/')[0] else 0
    result.append([item[0], label])
File_root.close()

with open('valid.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in result:
        writer.writerow(row)