import argparse, os, csv

def parse_args():
    '''
    Parse arguments
    '''

    parser = argparse.ArgumentParser(description="Convert dataset.cla file to dataset.csv file")
    parser.add_argument('-i', '--input', type=str, help="input cla file - cla_path")
    parser.add_argument('-o', '--output', type=str, help="output csv file - out_fn")
    
    return parser.parse_args()

def cla2csv(cla_path, out_fn):
    '''
    To convert the given .cla file into a workable .csv file
    `cla_path` (str): path to .cla file
    `out_fn` (str): path/filename of the output csv
    '''
    with open(out_fn, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["obj_id", "class_id"])
        with open(cla_path, 'r') as f:
            f.readline()
            class_num, obj_num = tuple(list(map(int, f.readline().split(" "))))
            for i in range(class_num):
                f.readline()
                obj_class_num = int(f.readline().split(" ")[-1])
                for j in range(obj_class_num):
                    obj_id = int(f.readline())
                    csv_writer.writerow([obj_id, i])
        
        # print(class_num, obj_num)

def main():
    args = parse_args()
    cla_path = args.input
    out_fn = args.output
    cla2csv(cla_path, out_fn)

main()
