# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse
from bleu import _bleu
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (line level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in json format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predictions, "r").readlines()
    gts = open(args.answers, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = len(gts)
    EM = 0.0
    preds = []
    gts = []
    for pred, gt in zip(preds, gts):
        pred = json.loads(pred)["code"]
        gt = json.loads(gt)["code"]
        preds.append(pred)
        gts.append(gt)
        if pred.split() == gt.split():
            EM += 1

    bleu_score = round(_bleu(gts, preds, subword_option=None)[0], 2)
    logger.info(f"BLEU: {bleu_score}, EM: {round(EM/total*100, 2)}")

    # try:
    #     os.remove("ground_truth.txt")
    # except Exception:
    #     pass

if __name__ == "__main__":
    main()