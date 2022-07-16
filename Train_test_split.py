from sklearn.model_selection import train_test_split
import csv
from collections import Counter

# Idea: load from csv, split train file and test file

def read_csv_data(src_dir):
    file = []
    label = []
    with open(src_dir, 'r', encoding='ascii') as f_in:
        csv_reader = csv.reader(f_in, delimiter=',')
        for row in csv_reader:
            file.append(row[0])
            label.append(row[1])
    f_in.close()
    return file, label

def split_data(X,Y, test_ratio=0.2, val_ratio=0.1):
    if test_ratio >0:
        X_p, X_test, Y_p, Y_test = train_test_split(X, Y, test_size=test_ratio, stratify=Y)
    else:
        X_p, Y_p = X, Y
    X_train, X_val, Y_train, Y_val = train_test_split(X_p, Y_p, test_size=val_ratio, stratify=Y_p)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def write_splitted_data(X,Y,trg_dir):
    with open(trg_dir,'w',encoding='ascii',newline='') as f_out:
        writer = csv.writer(f_out)
        for x,y in zip(X,Y):
            row = []
            row.append(x)
            row.append(y)
            writer.writerow(row)
    f_out.close()

if __name__ == '__main__':
    class_target = 'ostracods_genus'
    src_dir = class_target+'.csv'
    train_dir = './Metadata/'+class_target+'_train.csv'
    val_dir = './Metadata/'+class_target+'_val.csv'
    test_dir = './Metadata/'+class_target+'_test.csv'

    test_ratio = 0.1
    val_ratio = 0.1

    file, label = read_csv_data(src_dir)
    print(len(Counter(label).keys()))
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(file, label, test_ratio, val_ratio)
    write_splitted_data(X_train, Y_train, train_dir)
    write_splitted_data(X_val, Y_val, val_dir)
    write_splitted_data(X_test, Y_test, test_dir)




