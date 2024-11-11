import pandas as pd
import evaluate

# Constants
INPUT_CSV_FILENAME = './data/LED.csv'
OUTPUT_CSV_FILENAME = './metric_data/LED_huggingface.csv'

# Load the metrics
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')
bleu = evaluate.load('bleu')
# bleurt = evaluate.load('bleurt', module_type="metric")
meteor = evaluate.load('meteor')
chrf = evaluate.load('chrf')
perplexity = evaluate.load("perplexity", module_type="metric")


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

        # Compute various scores
        rouge_scores = rouge.compute(
            predictions=[pred_text], references=[gold_text])
        bert_scores = bertscore.compute(
            predictions=[pred_text], references=[gold_text], lang="en")
        bleu_scores = bleu.compute(
            predictions=[pred_text], references=[[gold_text]])
        # bleurt_scores = bleurt.compute(
        #    predictions=[pred_text], references=[gold_text])
        meteor_scores = meteor.compute(
            predictions=[pred_text], references=[gold_text])
        chrf_scores = chrf.compute(
            predictions=[pred_text], references=[[gold_text]])
        perplexity_scores = perplexity.compute(
            predictions=[pred_text], model_id='gpt2')

        metrics_results.append({
            'Gold': gold_text,
            'Predicted': pred_text,
            **rouge_scores,
            **bert_scores,
            **bleu_scores,
            # **bleurt_scores,
            **meteor_scores,
            **chrf_scores,
            **perplexity_scores
        })

    save_metrics_to_csv(metrics_results)


if __name__ == "__main__":
    main()
