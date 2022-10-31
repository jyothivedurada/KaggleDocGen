import sys
import pickle
import argparse
import re
import rouge

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def rouge_score(hyps, refs, ne):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                   max_n=4,
                   limit_length=True,
                   length_limit=100,
                   length_limit_type='words',
                   alpha=.5, # Default F1_score
                   weight_factor=1.2)

    scores = evaluator.get_scores(hyps, refs)
    return scores

def get_rouge_score(hyps, refs, ne):
    scores = rouge_score(hyps, refs, ne)
    return scores

def prepare_results_2(p, r, f, metric):
    return '{}: {}-{:5.2f}, {}-{:5.2f}, {}-{:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def RougeScoreBetween2Texts(pred, ref):
    scores = rouge_score([pred], [ref], False)
    result = []
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if metric in ["rouge-1"]:
            # result["{}(F1)".format(metric)] = "{:5.2f}".format(results['f'] * 100.0)
            result.append(prepare_results_2(results['p'], results['r'], results['f'], metric))
    return result

def print_scores(scores):
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(results['p'], results['r'], results['f'], metric))

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def RougeScoreBetween2Lists(predicts, targets):
    print("\nROUGE score: ")
    print_scores(get_rouge_score(predicts, targets, False))

def RougeScoreBetween2Files(predicts, targets):
    preds = dict()
    for c, line in enumerate(predicts):
        try:
            (fid, pred) = line.strip().split('\t')
        except ValueError:
            print("\nValue error while unpacking")
            print("\n", c, line)
            fid, pred = line.strip(), ""
        fid = int(fid)
        pred = pred.strip().split()
        pred = fil(pred)
        preds[fid] = ' '.join(pred)

    refs = list()
    newpreds = list()
    for line in targets:
        (fid, com) = line.strip().split('\t')
        fid = int(fid)
        com = com.strip().split()
        com = ' '.join(fil(com))
        
        if len(com) < 1:
            continue

        try:
            newpreds.append(preds[fid])
        except Exception as ex:
            print('no')
            continue
        
        refs.append(com)

    print("\nROUGE score: ")
    print_scores(get_rouge_score(newpreds, refs, False))

if __name__ == '__main__':
    
    # reference_file = '/home/cs19btech11056/cs21mtech12001-Tamal/HAConvGNN/repository/final_data/coms.test'
    # predictions_file = '/home/cs19btech11056/cs21mtech12001-Tamal/HAConvGNN/repository/modelout/predictions/predict_notebook.txt'
    reference_file = '/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_1.gold'
    predictions_file = '/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_1.output'

    print('preparing predictions list... ')
    preds = dict()
    predicts = open(predictions_file, 'r')
    for c, line in enumerate(predicts):
        try:
            (fid, pred) = line.strip().split('\t')
        except ValueError:
            print("\nValue error while unpacking")
            print("\n", c, line)
            fid, pred = line.strip(), ""
        fid = int(fid)
        pred = pred.strip().split()
        pred = fil(pred)
        preds[fid] = ' '.join(pred)
    predicts.close()

    refs = list()
    newpreds = list()
    targets = open(reference_file, 'r')
    for line in targets:
        (fid, com) = line.strip().split('\t')
        fid = int(fid)
        com = com.strip().split()
        com = ' '.join(fil(com))
        
        if len(com) < 1:
            continue

        try:
            newpreds.append(preds[fid])
        except Exception as ex:
            print('no')
            continue
        
        refs.append(com)

    print("\nSome prediction samples: ", newpreds[-5:])
    print("\nSome ground truth samples: ", refs[-5:])
    print('\nfinal status\n')
    print_scores(get_rouge_score(newpreds, refs, False))