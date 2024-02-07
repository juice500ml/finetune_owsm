import fire
import pandas as pd

zero_shot_langs = [
    "arta1239",
    "balk1252",
    "kach1280",
    "kaka1265",
    "kara1499",
    "mand1415",
    "nort2641",
    "pnar1238",
    "sanz1248",
    "sout2856",
    "sumb1241",
    "taba1259",
    "taul1251",
    "tehr1242",
    "trin1278",
]


def main(inf_results):
    df = pd.read_csv(inf_results)
    languages = sorted(df.lang.unique())
    results_seen = {}
    results_unseen = {}
    for lang in languages:
        langres = {
            "WER": df[df.lang == lang]["wer"].mean() * 100,
            "CER": df[df.lang == lang]["cer"].mean() * 100,
        }
        if lang not in zero_shot_langs:
            results_seen[lang] = langres
        else:
            results_unseen[lang] = langres

    print(pd.DataFrame.from_dict(results_seen, orient="index").sort_index().to_csv())
    print()
    print(pd.DataFrame.from_dict(results_unseen, orient="index").sort_index().to_csv())


if __name__ == "__main__":
    fire.Fire(main)
