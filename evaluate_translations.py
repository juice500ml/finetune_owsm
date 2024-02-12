import pandas as pd
import evaluate
import numpy as np
import fire

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


def main(preds):
    metrics = {
        "bleu": {
            "metric": evaluate.load("bleu"),
            "args": {},
            "result": "bleu",
            "aggregate": True,
        },
        "bleurt": {
            "metric": evaluate.load(
                "bleurt", module_type="metric", checkpoint="BLEURT-20"
            ),
            "args": {},
            "result": "scores",
            "aggregate": True,
        },
        "chrf++": {
            "metric": evaluate.load("chrf"),
            "args": {"word_order": 2},
            "result": "score",
            "aggregate": False,
        },
        # "bertscore": {
        #     "metric": evaluate.load("bertscore"),
        #     "args": {"lang": "en", "model_type": "distilbert-base-uncased"},
        #     "result": "f1",
        #     "aggregate": True,
        # },
    }
    task = "translation"
    preds = pd.read_csv(preds)

    preds = preds[preds["task"] == task]
    preds = preds[~preds["hyp"].isna()]
    languages = preds["lang"].unique()

    seen_scores = {}
    unseen_scores = {}
    for lang in languages:
        for metric, mval in metrics.items():
            subset = preds[preds["lang"] == lang]
            ref = subset["ref"]
            if metric == "bleu":
                ref = ref.apply(lambda x: [x])

            score = mval["metric"].compute(
                references=ref.to_list(),
                predictions=subset["hyp"].to_list(),
                **mval["args"],
            )[mval["result"]]

            if mval["aggregate"]:
                score = np.mean(score)
            if lang in zero_shot_langs:
                unseen_scores.setdefault(metric, []).append(score)
            else:
                seen_scores.setdefault(metric, []).append(score)

    for split, scores in [("Seen:", seen_scores), ("Unseen:", unseen_scores)]:
        print(split)
        for metric, vals in scores.items():
            print(f"{metric}: {np.mean(vals)}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
