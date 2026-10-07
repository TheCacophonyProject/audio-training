from pathlib import Path
import numpy as np
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels-key", default="ebird_labels", help="Key of labels in metadata "
    )
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
    first_none_index = -1
    second_none_index = -1
    first_labels = None
    second_labels = None
    if Path(args.first_confusion).suffix == ".npz":
        data = np.load(args.first_confusion)
        first_labels = list(data["labels"])
        if "None" in first_labels:
            first_none_index = first_labels.index("None")
        else:
            first_none_index = first_labels.index("nothing")
        first_cm = data["cm"]
    else:
        first_cm = np.load(args.first_confusion)

    if Path(args.second_confusion).suffix == ".npz":
        data = np.load(args.second_confusion)
        second_labels = list(data["labels"])
        if "None" in second_labels:
            second_none_index = second_labels.index("None")
        else:
            second_none_index = second_labels.index("nothing")
        second_cm = data["cm"]
    else:
        second_cm = np.load(args.second_confusion)

    if first_labels is None:

        first_cm_meta_file = args.first_confusion.parent / "metadata.txt"
        print("Loading meta from ", first_cm_meta_file)
        with first_cm_meta_file.open("r") as f:
            first_meta = json.load(f)
        first_labels = first_meta[args.labels_key]

    if second_labels is None:
        second_cm_meta_file = args.second_confusion.parent / "metadata.txt"
        print("Loading meta from ", second_cm_meta_file)
        with second_cm_meta_file.open("r") as f:
            second_meta = json.load(f)
        second_labels = second_meta[args.labels_key]
    pre_labels = ["bird", "human", "noise"]

    print("Comparing confusions ", first_labels, " vs ", second_labels)
    print(len(first_labels), "len", len(second_labels))
    incorrect_score = 0
    first_inccorect = 0
    second_incorrect = 0
    total = 0

    for label in first_labels:
        if label not in second_labels:
            print("First label has ", label, " second does not")

    for label in second_labels:
        if label not in first_labels:
            print("Second label has ", label, " first does not")

    total_samples = 0
    first_correct = 0
    second_correct = 0
    second_total_samples = 0
    first_pre_lbl_error = 0
    second_pre_lbl_error = 0
    first_none_total = 0
    second_none_total = 0
    for i, label in enumerate(first_labels):
        if i >= len(first_cm):
            break
        first_count = first_cm[i][i]
        first_none = first_cm[i][first_none_index]
        first_total = np.sum(first_cm[i])
        first_none_total += first_none
        label_total = np.sum(first_cm[i])
        total_samples += label_total
        first_correct += first_count

        row_copy = first_cm[i].copy()
        
        row_copy[i] = 0
        row_copy[first_none_index] = 0
        most_wrong = np.argmax(row_copy)
        # print(label,first_cm[i])
        if label in second_labels:
            second_i = second_labels.index(label)

            second_count = second_cm[second_i][second_i]
            second_correct += second_count
            second_none = second_cm[second_i][second_none_index]
            second_total = np.sum(second_cm[second_i])
            second_none_total += second_none

            row_copy = second_cm[second_i].copy()
      
          
            row_copy[second_i] = 0
            row_copy[second_none_index] = 0
            second_most_wrong = np.argmax(row_copy)

            if second_total != first_total:
                print(f"{label} First total is {first_total} second is {second_total}")
            # assert (
          
            first_inccorect += first_total - first_count - first_none 

            second_total_samples += second_total
            second_incorrect += second_total - second_count - second_none 
           
            if first_total == 0:
                first_acc = 0
                first_none = 0
                first_wrong_acc = 0
            else:
                first_acc = round(100 * first_count / first_total)
                first_none = round(100 * first_none / first_total)
                first_wrong_acc = round(first_cm[i][most_wrong] / first_total * 100)

            if second_total == 0:
                second_acc = 0
                second_none = 0
                second_wrong_acc = 0
            else:
                second_acc = round(100 * second_count / second_total)
                second_none = round(100 * second_none / second_total)
                second_wrong_acc = round(
                    second_cm[second_i][second_most_wrong] / second_total * 100
                )

            print(
                f"For {label}:  {first_total}#  Correct diff:  {first_count-second_count}#, Accuracies  {first_acc}% vs {second_acc}%,  Percent None: {first_none} vs {second_none}, Animal most incorrect {first_wrong_acc} % ( {first_labels[most_wrong]} #), {first_cm[i][most_wrong]}# and second most wrong {second_labels[second_most_wrong]}, {second_wrong_acc}% ( {second_cm[second_i][second_most_wrong]} #)"
            )
            total += first_count - second_count

        else:
            print(f"Label {label} only in first")
  

    print(
        f"Total diff is {total} ( {round(100* total/ total_samples,1)}) first incorrect {first_inccorect} {round(100*first_inccorect / total_samples,1) }% second incorrect {second_incorrect} {round(100*second_incorrect/second_total_samples,1)}% score diff {round(100* (first_inccorect - second_incorrect) / total_samples,1)}"
    )

    acc_percent = abs(total / total_samples)
    inc_percent = abs((first_inccorect - second_incorrect) / total_samples)
    diff = acc_percent - inc_percent
    print("Acc - Inc", round(100 * diff, 2))
    print("Total samples are ", total_samples)
    print(
        f"First: {first_correct} / {total_samples} = ",
        first_correct / total_samples,
        f" vs Second: {second_correct} / {second_total_samples} = ",
        second_correct / second_total_samples,
    )
    if total > 0:
        print("Better model is first ", args.first_confusion)
    else:
        print("Better model is second ", args.second_confusion)
    print("Pre lbl error ", first_pre_lbl_error, " second ", second_pre_lbl_error)

    # animal error = anything not correct and not None (includes predicted as bird)
    first_error = total_samples - first_correct - first_none_total
    second_error = second_total_samples - second_correct - second_none_total
    print(
        f"None: first {first_none_total} {round(100 * first_none_total / total_samples, 1)}% vs second {second_none_total} {round(100 * second_none_total / second_total_samples, 1)}%"
    )
    print(
        f"Animal error: first {first_error} {round(100 * first_error / total_samples, 1)}% vs second {second_error} {round(100 * second_error / second_total_samples, 1)}%"
    )
    print(
        f"Precision (correct / not None): first {round(100 * first_correct / (first_correct + first_error), 1)}% vs second {round(100 * second_correct / (second_correct + second_error), 1)}%"
    )


if __name__ == "__main__":
    main()
