import pandas as pd
from collections import defaultdict
from typing import List
import numpy as np

# from rouge import Rouge
# from bert_score import score
from moverscore_v2 import get_idf_dict, word_mover_score

# Constants
INPUT_CSV_FILENAME = '.\\data\\LED.csv'
OUTPUT_CSV_FILENAME = '.\\metric_data\\LED_metric.csv'
LANG = 'en'


def sentence_score(hypothesis: str, references: List[str]):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)
    scores = word_mover_score(references, hypothesis, idf_dict_ref,
                              idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)

    return np.mean(scores)


def get_mover_score(gold_text, pred_text):
    gold_sentences = gold_text.split('. ')
    pred_sentences = pred_text.split('. ')
    idf_dict_hyp = get_idf_dict(pred_sentences)
    idf_dict_ref = get_idf_dict(gold_sentences)
    return sentence_score(pred_text, [gold_text])


def read_data(filename):
    return pd.read_csv(filename, sep=';')


# Commenting out the bert and rouge precomputation function
# def precompute_scores(gold_texts, pred_texts):
#     _, _, bert_f1_list = score(pred_texts, gold_texts, lang=LANG)
#     bert_f1_list = bert_f1_list.tolist()

#     rouge = Rouge()
#     rouge_scores_list = rouge.get_scores(pred_texts, gold_texts)

#     return bert_f1_list, rouge_scores_list

def save_metrics_to_csv(metrics_results):
    df_metrics = pd.DataFrame(metrics_results)
    df_metrics.to_csv(OUTPUT_CSV_FILENAME, index=False)


def main():
    df = read_data(INPUT_CSV_FILENAME)
    gold_texts = df['Gold'].tolist()
    pred_texts = df['Predicted'].tolist()

    # Commenting out the bert and rouge precomputation
    # bert_f1_list, rouge_scores_list = precompute_scores(gold_texts, pred_texts)

    metrics_results = []
    for i in range(len(gold_texts)):
        # Commenting out the calculation of other metrics
        # metrics = calculate_metrics(
        #     gold_texts[i], pred_texts[i], bert_f1_list[i], rouge_scores_list[i])

        mover_score = get_mover_score(gold_texts[i], pred_texts[i])

        metrics_results.append({
            'Gold': gold_texts[i],
            'Predicted': pred_texts[i],
            'mover_score': mover_score
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
