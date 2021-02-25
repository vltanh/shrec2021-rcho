import argparse
import os
import csv


def parse_args():
    '''
    Parse arguments
    '''
    parser = argparse.ArgumentParser(
        description="To generate a CSV files containing query objects id"
    )
    parser.add_argument('-s', '--source', type=str,
                        help="directory containing query objects - objs_dir")

    return parser.parse_args()


def generate_csv_from_dir(objs_dir, out_fn):
    '''
    `objs_dir` (str): directory containing query objects
    `out_fn` (str): path/filename of the output csv
    '''
    with open(out_fn, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["obj_id"])
        for obj in sorted(os.listdir(objs_dir)):
            if obj.endswith(".obj"):
                obj_id = obj.split(".")[0]
                csv_writer.writerow([obj_id])


if __name__ == '__main__':
    args = parse_args()
    objs_dir = args.source
    generate_csv_from_dir(objs_dir, "test.csv")


main()
