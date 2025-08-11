from pathlib import Path
import numpy as np
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "first_confusion",
        help="First confusion to compare",
    )

    parser.add_argument("second_confusion", help="Second confusion to compare")
    args = parser.parse_args()
    args.first_confusion = Path(args.first_confusion)

    args.second_confusion = Path(args.second_confusion)
    return args


def main():
    args = parse_args()
    first_cm = np.load(args.first_confusion)
    second_cm = np.load(args.second_confusion)

    first_cm_meta_file = args.first_confusion.parent / "metadata.txt"
    print("Loading meta from ", first_cm_meta_file)
    with first_cm_meta_file.open("r") as f:
        first_meta = json.load(f)
    second_cm_meta_file = args.second_confusion.parent / "metadata.txt"
    print("Loading meta from ", second_cm_meta_file)
    with second_cm_meta_file.open("r") as f:
        second_meta = json.load(f)

    first_labels = first_meta["labels"]
    second_labels = second_meta["labels"]
    print("Comparing confusions ", first_labels, " vs ", second_labels)
    total = 0
    for i, label in enumerate(first_labels):
        first_acc = first_cm[i][i]
        first_none = first_cm[i][-1]
        first_total = np.sum(first_cm[i])
        if label in second_labels:
            second_i = second_labels.index(label)
            second_acc = second_cm[second_i][second_i]
            second_none = second_cm[second_i][-1]
            second_total = np.sum(second_cm[second_i])
            assert second_total == first_total
            first_acc = round(100 * first_acc / first_total)

            second_acc = round(100 * second_acc / second_total)
            print(
                f"For {label} have {first_acc-second_acc} from  {first_acc} vs {second_acc} None accuracies are {round(100*first_none/first_total)} vs {round(100*second_none/second_total)} total is {first_total} "
            )
            total += first_acc - second_acc
        else:
            print(f"Label {label} only in first")
    print("Total diff is ", total)


if __name__ == "__main__":
    main()
