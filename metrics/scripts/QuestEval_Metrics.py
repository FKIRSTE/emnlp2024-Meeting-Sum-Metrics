import pandas as pd
from questeval.questeval_metric import QuestEval

# Constants
INPUT_CSV_FILENAME = './data/LED.csv'
OUTPUT_CSV_FILENAME = './metric_data/LED_questeval.csv'

# Initialize QuestEval with do_weighter=True
questeval = QuestEval(do_weighter=True)


def read_data(filename):
    return pd.read_csv(filename, sep=';')


def save_metrics_to_csv(metrics_results):
    df_metrics = pd.DataFrame(metrics_results)
    df_metrics.to_csv(OUTPUT_CSV_FILENAME, index=False)


def main():
    df = read_data(INPUT_CSV_FILENAME)
    gold_texts = df['Gold'].tolist()
    pred_texts = df['Predicted'].tolist()
    # Will use None if 'Reference' column is not in the CSV
    ref_texts = df.get('Transcript', [None]*len(gold_texts)).tolist()

    metrics_results = []

    for i in range(len(gold_texts)):
        gold_text = gold_texts[i]
        pred_text = pred_texts[i]
        ref_text = ref_texts[i]

        if ref_text:
            # Compute QuestEval scores with reference
            questeval_scores = questeval.compute_all(
                pred_text, gold_text, ref_text)
        else:
            # Compute QuestEval scores without reference
            questeval_scores = questeval.compute_all(pred_text, gold_text)

        metrics_results.append({
            'Gold': gold_text,
            'Predicted': pred_text,
            'QuestEval_FScore': questeval_scores['scores']['fscore'],
            'QuestEval_Precision': questeval_scores['scores']['precision'],
            'QuestEval_Recall': questeval_scores['scores']['recall']
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
