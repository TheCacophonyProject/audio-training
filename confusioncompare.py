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

    first_labels = first_meta["ebird_labels"]
    # re_l = first_meta["remapped_labels"]
    # for k, v in re_l.items():
    #     mapped_lbl = first_labels[v]
    #     if mapped_lbl != k:
    #         print("Mapped is ",mapped_lbl, k)
    #         if k in first_labels:
    #             first_labels.remove(k)
    #             print("Remvoing", k)
    second_labels = second_meta["ebird_labels"]

    pre_labels = ["bird", "human", "noise"]

    if len(first_cm[0]) != len(first_labels) + 1:
        first_labels.extend(pre_labels)
    if len(second_cm[0]) != len(second_labels) + 1:

        second_labels.extend(pre_labels)

    print(second_labels)
    print("Comparing confusions ", first_labels, " vs ", second_labels)
    print(len(first_labels), len(second_labels))
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
    # return
    # due to a bug some cms have None twice as last columns but last column should have the actual percentages
    # assert len(first_cm) == len(first_labels)+1, f"First cm is len {len(first_cm)} while labels {len(first_labels)}"
    # assert len(second_cm) == len(second_labels)+1, f"Second cm is len {len(second_cm)} while labels {len(second_labels)}"
    # morepork_i = first_labels.index("morepo2")
    # noise_i = second_labels.index("noise")
    # bird_i = second_labels.index("bird")
    # animal_i = second_labels.index("animal")
    # human_i = second_labels.index("human")

    # for i, label in enumerate(first_labels):
    #     first_row = first_cm[i]
    #     second_row = second_cm[i]

    #     # second_row[-1] =first_row[-1]
    #     more_diff = second_row[morepork_i] - first_row[morepork_i]
    #     # second_row[-1] += more_diff + second_row[noise_i]+ second_row[bird_i] + second_row[human_i]
    #     # second_row[-1]+=more_diff
    #     # second_row[-1]+=second_row[noise_i]

    #     # second_row[-1]+=second_row[noise_i]
    #     # second_row[noise_i] = 0
    #     # second_row[bird_i] = 0
    #     # second_row[animal_i] = 0
    #     # second_row[human_i]=0

    #     print("more diff is ",more_diff)
    #     # second_row[morepork_i] = first_row[morepork_i]
    # print("Ignoring moreporks from pr emodel")
    total_samples = 0
    first_correct = 0
    second_correct = 0
    second_total_samples = 0
    first_pre_lbl_error = 0
    second_pre_lbl_error = 0

    for i, label in enumerate(first_labels):
        # if label in pre_labels:
        # continue
        if label in ["human", "morepo2"]:
            continue
        first_count = first_cm[i][i]
        first_none = first_cm[i][-1]
        first_total = np.sum(first_cm[i])
        label_total = np.sum(first_cm[i])
        total_samples += label_total
        first_correct += first_count

        row_copy = first_cm[i].copy()
        first_bird_c = 0
        if "bird" in first_labels:
            first_bird_c = first_cm[i][first_labels.index("bird")]
            row_copy[first_labels.index("bird")] = 0

        if label == "noise":
            row_copy[first_labels.index("insect")] = 0
        if label != "insect" and label not in pre_labels:
            for pre_l in pre_labels:
                if pre_l == "bird":
                    continue
                if pre_l in first_labels:
                    pre_i = first_labels.index(pre_l)
                    first_pre_lbl_error += first_cm[i][pre_i]
            # print("Adding error for ",label,first_pre_lbl_error,first_cm[i],np.sum(first_cm[i]))
        row_copy[i] = 0
        row_copy[-1] = 0
        most_wrong = np.argmax(row_copy)
        # print(label,first_cm[i])
        if label in second_labels:
            second_i = second_labels.index(label)
            second_count = second_cm[second_i][second_i]
            second_correct += second_count
            second_none = second_cm[second_i][-1]
            second_total = np.sum(second_cm[second_i])

            # if label != "insect" and label not in pre_labels:
            #     for pre_l in pre_labels:
            #         if pre_l == "bird":
            #             continue
            #         if pre_l in second_labels:
            #             pre_i = second_labels.index(pre_l)
            #             second_pre_lbl_error += second_cm[i][pre_i]

            row_copy = second_cm[second_i].copy()
            if "bird" in second_labels:
                row_copy[second_labels.index("bird")] = 0

            if label == "noise":
                row_copy[second_labels.index("insect")] = 0
            row_copy[second_i] = 0
            row_copy[-1] = 0
            second_most_wrong = np.argmax(row_copy)

            # if second_total != first_total:
            # print(f"{label} First total is {first_total} second is {second_total}")
            # assert second_total == first_total, f"{label} First total is {first_total} second is {second_total}"
            # if first_total == 0:
            #     continue
            bird_c = 0
            if "bird" in second_labels:
                bird_c = second_cm[second_i][second_labels.index("bird")]

            if label in pre_labels:
                first_bird_c = 0
                bird_c = 0
            first_inccorect += first_total - first_count - first_none - first_bird_c
            # bird_c = 0
            second_total_samples += second_total
            second_incorrect += second_total - second_count - second_none - bird_c
            # assert first_inccorect == second_incorrect, f"{first_cm[i] } vs {second_cm[second_i]}"
            # print("for first",first_total - first_count - first_none," Second ",second_total - second_count - second_none-bird_c)
            # print("tally is ",second_incorrect)
            # print(
            #     second_total,
            #     "correct ",
            #     second_count,
            #     " none ",
            #     second_none,
            #     second_total - second_count - second_none,
            # )
            # print(first_cm[i],second_cm[i])
            # print(second_count,second_total)
            if first_total == 0:
                first_acc = 0
                first_none = 0
            else:
                first_acc = round(100 * first_count / first_total)
                first_none = round(100 * first_none / first_total)
            if second_total == 0:
                second_acc = 0
                second_none = 0
            else:
                second_acc = round(100 * second_count / second_total)
                second_none = round(100 * second_none / second_total)
            print(
                f"For {label} have {first_count-second_count} samples diff from  {first_acc}% vs {second_acc}% None accuracies are {first_none} vs {second_none} most wrong {first_labels[most_wrong]}, {first_cm[i][most_wrong]}# and second most wrong {second_labels[second_most_wrong]}, {second_cm[second_i][second_most_wrong]}# total is {first_total} "
            )
            total += first_count - second_count

        else:
            print(f"Label {label} only in first")
    # second_correct = 0
    # second_total_samples = 0
    # second_incorrect = 0
    # for second_i, label in enumerate(second_labels):
    #     second_count = second_cm[second_i][second_i]
    #     second_correct += second_count
    #     second_total = np.sum(second_cm[second_i])
    #     second_none = second_cm[second_i][-1]

    #     second_total_samples += second_total
    #     second_incorrect += second_total - second_count - second_none

    print(
        f"Total diff is {total} ( {round(100* total/ total_samples,1)}) first inccorect {first_inccorect} second incorrect {second_incorrect} score diff {round(100* (first_inccorect - second_incorrect) / total_samples,1)}"
    )

    acc_percent = abs(total / total_samples)
    inc_percent = abs((first_inccorect - second_incorrect) / total_samples)
    diff = acc_percent - inc_percent
    print("Acc - Inc", round(100 * diff, 2))
    print("Total samples are ", total_samples)
    print(
        f"{first_correct} / {total_samples} = ",
        first_correct / total_samples,
        f" vs {second_correct} / {second_total_samples} = ",
        second_correct / second_total_samples,
    )
    if total > 0:
        print("Better model is first ", args.first_confusion)
    else:
        print("Better model is second ", args.second_confusion)
    print("Pre lbl error ", first_pre_lbl_error, " second ", second_pre_lbl_error)


if __name__ == "__main__":
    main()
