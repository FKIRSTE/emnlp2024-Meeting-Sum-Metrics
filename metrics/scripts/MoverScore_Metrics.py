import pandas as pd
from collections import defaultdict
from typing import List, Union, Iterable
from itertools import zip_longest
import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score

# Constants
INPUT_CSV_FILENAME = './data/LED.csv'
OUTPUT_CSV_FILENAME = './metric_data/LED_moverscore.csv'


def sentence_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)
    sentence_score = 0
    scores = word_mover_score(references, hypothesis, idf_dict_ref,
                              idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    sentence_score = np.mean(scores)
    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score


def corpus_score(sys_stream: List[str],
                 ref_streams: Union[str, List[Iterable[str]]], trace=0):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError(
                "Source and reference streams have different lengths!")
        hypo, *refs = lines
        corpus_score += sentence_score(hypo, refs, trace=0)
    corpus_score /= len(sys_stream)
    return corpus_score


def read_data(filename):
    return pd.read_csv(filename, sep=';')


def save_metrics_to_csv(metrics_results):
    df_metrics = pd.DataFrame(metrics_results)
    df_metrics.to_csv(OUTPUT_CSV_FILENAME, index=False)


def main():
    df = read_data(INPUT_CSV_FILENAME)
    transcripts = df['Transcript'].tolist()
    gold_texts = df['Gold'].tolist()
    pred_texts = df['Predicted'].tolist()

    metrics_results = []

    for i in range(len(transcripts)):
        transcript = transcripts[i]
        gold_text = gold_texts[i].split('. ')  # Splitting by sentence
        pred_text = pred_texts[i].split('. ')  # Splitting by sentence

        # Join the lists back into strings
        gold_text_str = '. '.join(gold_text)
        pred_text_str = '. '.join(pred_text)

        # Compute MoverScore at corpus level
        moverscore = corpus_score([pred_text_str], [gold_text_str])

        metrics_results.append({
            'Transcript': transcript,
            # Keeping original, unsplit text for readability
            'Gold': gold_texts[i],
            # Keeping original, unsplit text for readability
            'Predicted': pred_texts[i],
            'MoverScore': moverscore
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
