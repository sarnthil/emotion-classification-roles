import glob
from itertools import groupby

results = sorted(glob.glob("../../results/eval*.txt"))


def kf_dataset(name):
    # eval-unified_eca_inband_cause-1603661511.txt
    pre, dataset, mark_type, *rest = name.split("-")[-2].split("_")
    return dataset


def kf_mark_type(name):
    pre, dataset, mark_type, *rest = name.split("-")[-2].split("_")
    return mark_type


clean = []
for dataset, ds_results in groupby(sorted(results, key=kf_dataset), kf_dataset):
    for mark_type, mt_results in groupby(
        sorted(ds_results, key=kf_mark_type), kf_mark_type
    ):
        for result in mt_results:
            with open(result) as fr:
                if mark_type != "all":
                    role = result.split("_")[-1].split("-")[0]
                else:
                    role = None
                next(fr)
                next(fr)
                line = next(fr)
                f1 = line.split("=")[1]
                print(f"{dataset}\t{mark_type}\t{role}\t{float(f1):.2f}")
