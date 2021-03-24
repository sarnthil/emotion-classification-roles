import json

roles = [role for role in ["iob-cause", "iob-cue", "iob-target", "iob-experiencer"]]
datasets = ["emotion-stimulus", "electoral_tweets", "eca", "gne", "reman"]
dsnames = {"emotion-stimulus": "ES", "electoral_tweets": "ET"}

with open("workdata/position-experiment/results.json") as f:
    data = json.load(f)

for dataset in datasets:
    emotions = sorted({key for setting in data[dataset] for key in data[dataset][setting] if key.lower() not in ["love"] and not any(c in key for c in "-_")})
    print(r"  \multirow{%r}{*}{%s}" % (len(emotions), dsnames.get(dataset, dataset).upper()))
    for emotion in emotions:
        print(" &", emotion.title(), end="")
        for setting in roles:
            for key in ("precision", "recall", "f1"):
                val = data[dataset].get(setting, {}).get(emotion, {}).get(key, None)
                print(" &", f"{val:.2f}" if val is not None else "--", end="")
        print(r"\\")
    print(r"\cmidrule(r{7mm}){1-2}\cmidrule(r{10mm}l){3-5}\cmidrule{6-8}\cmidrule{9-11}\cmidrule{12-14}\cmidrule{15-17}")
