"""Metrics computation for VLM benchmarking."""

import numpy as np
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class MetricsComputer:
    """Compute all evaluation metrics."""

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def compute_all(self, predictions, references, pred_answers, gt_answers):
        """Compute all metrics."""
        results = {}

        # MCQ Accuracy
        correct = sum(1 for p, g in zip(pred_answers, gt_answers) if p == g)
        results['mcq_accuracy'] = correct / len(gt_answers) if gt_answers else 0.0

        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        meteor_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # BLEU
            if pred_tokens and ref_tokens:
                bleu = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=self.smoothing
                )
            else:
                bleu = 0.0
            bleu_scores.append(bleu)

            # ROUGE
            rouge_result = self.rouge.score(ref, pred)
            rouge1_scores.append(rouge_result['rouge1'].fmeasure)
            rouge2_scores.append(rouge_result['rouge2'].fmeasure)
            rougeL_scores.append(rouge_result['rougeL'].fmeasure)

            # METEOR
            try:
                from nltk.translate.meteor_score import meteor_score
                m = meteor_score([ref_tokens], pred_tokens)
            except Exception:
                m = 0.0
            meteor_scores.append(m)

        results['bleu'] = np.mean(bleu_scores)
        results['rouge1'] = np.mean(rouge1_scores)
        results['rouge2'] = np.mean(rouge2_scores)
        results['rougeL'] = np.mean(rougeL_scores)
        results['meteor'] = np.mean(meteor_scores)

        # BERTScore
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                predictions, references,
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en", verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            results['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            print(f"  [BERTScore Error]: {e}")
            results['bertscore_precision'] = 0.0
            results['bertscore_recall'] = 0.0
            results['bertscore_f1'] = 0.0

        return results

    def compute_per_task(self, df):
        """Compute metrics grouped by task type."""
        task_results = {}
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            correct = (task_df['pred_answer'] == task_df['gt_answer']).sum()
            task_results[task] = {
                'accuracy': correct / len(task_df) if len(task_df) > 0 else 0.0,
                'count': len(task_df)
            }
        return task_results
