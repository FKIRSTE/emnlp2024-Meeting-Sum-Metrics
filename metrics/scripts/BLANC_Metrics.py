import pandas as pd
from blanc import BlancHelp, BlancTune

# Constants
INPUT_CSV_FILENAME = './data/LED.csv'
OUTPUT_CSV_FILENAME = './metric_data/LED_blanc.csv'

# Initialize the BLANC metric with GPU support
blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
blanc_tune = BlancTune(device='cuda', inference_batch_size=24,
                       finetune_mask_evenly=False, finetune_batch_size=24)


def read_data(filename):
    return pd.read_csv(filename, sep=';')


def save_metrics_to_csv(metrics_results):
    df_metrics = pd.DataFrame(metrics_results)
    df_metrics.to_csv(OUTPUT_CSV_FILENAME, index=False)


def main():
    df = read_data(INPUT_CSV_FILENAME)
    gold_texts = df['Gold'].tolist()
    pred_texts = df['Predicted'].tolist()

    metrics_results = []

    for i in range(len(gold_texts)):
        gold_text = gold_texts[i]
        pred_text = pred_texts[i]

        # Compute BLANC scores
        blanc_help_score = blanc_help.eval_once(gold_text, pred_text)
        blanc_tune_score = blanc_tune.eval_once(gold_text, pred_text)

        metrics_results.append({
            'Gold': gold_text,
            'Predicted': pred_text,
            'blanc_help_score': blanc_help_score,
            'blanc_tune_score': blanc_tune_score,
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
