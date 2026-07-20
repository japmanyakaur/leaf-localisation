"""
Collapse PlantDoc's 29 species/disease classes into a single 'leaf' class (id 0).

We only care about *where* leaves are, not which species/disease they are,
so every box's class id gets rewritten to 0.

Images are copied from the raw dataset into data/processed/ unchanged (only
the label .txt files are actually rewritten). We use a plain file copy rather
than a symlink here because Windows restricts symlink creation to admin users
by default, which would make this script fail for most people running it locally.
"""
import os
import shutil

RAW = r"D:\UserData\Downloads\yolo_dataset"
OUT = "data/processed"
SPLITS = ["train", "val", "test"]


def relabel_file(src_path, dst_path):
    with open(src_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        parts[0] = "0"  # force every class id to 0 = 'leaf'
        new_lines.append(" ".join(parts))

    with open(dst_path, "w") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))


def main():
    for split in SPLITS:
        src_images = os.path.join(RAW, split, "images")
        src_labels = os.path.join(RAW, split, "labels")
        dst_images = os.path.join(OUT, split, "images")
        dst_labels = os.path.join(OUT, split, "labels")
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        # copy images (Windows symlinks need admin rights, so we copy instead)
        n_img = 0
        for name in os.listdir(src_images):
            src = os.path.join(src_images, name)
            dst = os.path.join(dst_images, name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            n_img += 1

        # rewrite labels with class id forced to 0
        n_lbl = 0
        for name in os.listdir(src_labels):
            relabel_file(os.path.join(src_labels, name), os.path.join(dst_labels, name))
            n_lbl += 1

        print(f"{split}: {n_img} images copied, {n_lbl} labels relabeled")


if __name__ == "__main__":
    main()
