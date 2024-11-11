import pandas as pd
from lens.lens_score import LENS

# Constants
INPUT_CSV_FILENAME = './data/LED.csv'
OUTPUT_CSV_FILENAME = './metric_data/LED_lens.csv'
LENS_MODEL_PATH = "./LENS-checkpoint/LENS/checkpoints/epoch=5-step=6102.ckpt"

# Initialize LENS
lens_metric = LENS(LENS_MODEL_PATH, rescale=True)


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
        gold_text = gold_texts[i]
        pred_text = pred_texts[i]

        # Compute LENS scores
        lens_scores = lens_metric.score([transcript], [pred_text], [
                                        [gold_text]], batch_size=8, gpus=1)

        metrics_results.append({
            'Transcript': transcript,
            'Gold': gold_text,
            'Predicted': pred_text,
            'LENS_Score': lens_scores[0]
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
