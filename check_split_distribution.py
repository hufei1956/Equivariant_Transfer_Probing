
import json
import os
import collections

def check_split():
    split_file = "splits/dataset-resized_split.json"
    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return

    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Classes: {data['classes']}")
    
    splits = ["train", "val", "test"]
    for s in splits:
        print(f"\n--- {s.upper()} set ---")
        paths = data[s]
        # path format: "class_name/filename.jpg"
        labels = [p.split("/")[0] for p in paths]
        counts = collections.Counter(labels)
        total = len(paths)
        print(f"Total images: {total}")
        for cls in data['classes']:
            c = counts.get(cls, 0)
            ratio = c / total if total > 0 else 0
            print(f"  {cls}: {c} ({ratio:.2%})")

if __name__ == "__main__":
    check_split()
