import pandas as pd

from training.prepare_data import add_validation, drop_duplicates


def _df(rows):
    return pd.DataFrame(rows, columns=["path", "label", "split", "md5"])


def test_drop_duplicates_keeps_test_copy():
    df = _df(
        [
            ("train/benign/a.jpg", "benign", "train", "x"),
            ("test/benign/b.jpg", "benign", "test", "x"),
            ("train/benign/c.jpg", "benign", "train", "y"),
        ]
    )
    out = drop_duplicates(df)
    assert sorted(out["path"]) == ["test/benign/b.jpg", "train/benign/c.jpg"]


def test_add_validation_only_moves_train_and_is_stratified():
    rows = [
        (f"train/{lbl}/{i}.jpg", lbl, "train", f"{lbl}{i}")
        for lbl in ("benign", "malignant")
        for i in range(20)
    ]
    rows += [("test/benign/t.jpg", "benign", "test", "t")]
    out = add_validation(_df(rows), val_size=0.2)
    val = out[out["split"] == "val"]
    assert val["label"].value_counts().to_dict() == {"benign": 4, "malignant": 4}
    assert (out["split"] == "test").sum() == 1
