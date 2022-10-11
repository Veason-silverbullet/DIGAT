import os
import math


class Criteria:
    def __init__(self, run_index, auc, mrr, ndcg5, ndcg10):
        self.run_index = run_index
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10

    def __gt__(self, value):
        return self.run_index > value.run_index

    def __ge__(self, value):
        return self.run_index >= value.run_index

    def __lt__(self, value):
        return self.run_index < value.run_index

    def __le__(self, value):
        return self.run_index <= value.run_index

    def __str__(self):
        return '#%d\t%.4f\t%.4f\t%.4f\t%.4f' % (self.run_index, self.auc, self.mrr, self.ndcg5, self.ndcg10)

class OverallCriteria:
    def __init__(self, model_name, auc, mrr, ndcg5, ndcg10):
        self.model_name = model_name
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10

    def __gt__(self, value):
        return self.model_name > value.model_name

    def __ge__(self, value):
        return self.model_name >= value.model_name

    def __lt__(self, value):
        return self.model_name < value.model_name

    def __le__(self, value):
        return self.model_name <= value.model_name

    def __str__(self):
        return '%s\t%.4f\t%.4f\t%.4f\t%.4f' % (self.model_name, self.auc, self.mrr, self.ndcg5, self.ndcg10)

def aggregate_criteria(criteria_list, experiment_results_f):
    sum_auc = 0
    sum_mrr = 0
    sum_ndcg5 = 0
    sum_ndcg10 = 0
    std_auc = 0
    std_mrr = 0
    std_ndcg5 = 0
    std_ndcg10 = 0
    N = len(criteria_list)
    assert N > 0
    for criteria in criteria_list:
        sum_auc += criteria.auc
        sum_mrr += criteria.mrr
        sum_ndcg5 += criteria.ndcg5
        sum_ndcg10 += criteria.ndcg10
    mean_auc = sum_auc / N
    mean_mrr = sum_mrr / N
    mean_ndcg5 = sum_ndcg5 / N
    mean_ndcg10 = sum_ndcg10 / N
    for criteria in criteria_list:
        std_auc += (criteria.auc - mean_auc) ** 2
        std_mrr += (criteria.mrr - mean_mrr) ** 2
        std_ndcg5 += (criteria.ndcg5 - mean_ndcg5) ** 2
        std_ndcg10 += (criteria.ndcg10 - mean_ndcg10) ** 2
    std_auc = math.sqrt(std_auc / N)
    std_mrr = math.sqrt(std_mrr / N)
    std_ndcg5 = math.sqrt(std_ndcg5 / N)
    std_ndcg10 = math.sqrt(std_ndcg10 / N)
    experiment_results_f.write('\nAvg\t%.4f\t%.4f\t%.4f\t%.4f\n' % (mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))
    experiment_results_f.write('Std\t%.4f\t%.4f\t%.4f\t%.4f\n' % (std_auc, std_mrr, std_ndcg5, std_ndcg10))
    return mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10

def aggregate_dev_result(dataset: str):
    assert dataset in ['MIND-small', 'MIND-large'], 'Dataset is chosen from \'MIND-small\' and \'MIND-large\''
    if os.path.exists('results/' + dataset):
        for sub_dir in os.listdir('results/' + dataset):
            if sub_dir in ['NRMS', 'NRMS-SA']:
                with open('results/' + dataset + '/' + sub_dir + '/experiment_results-dev.tsv', 'w', encoding='utf-8') as experiment_results_f:
                    experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                    criteria_list = []
                    for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                        if result_file[0] == '#' and result_file[-4:] == '-dev':
                            with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                line = result_f.read()
                                if len(line.strip()) != 0:
                                    run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                    criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                    if len(criteria_list) > 0:
                        criteria_list.sort()
                        for criteria in criteria_list:
                            experiment_results_f.write(str(criteria) + '\n')
                        aggregate_criteria(criteria_list, experiment_results_f)

def aggregate_test_result(dataset: str):
    assert dataset in ['MIND-small', 'MIND-large'], 'Dataset is chosen from \'MIND-small\' and \'MIND-large\''
    if os.path.exists('results/' + dataset):
        with open('results/%s/overall.tsv' % dataset, 'w', encoding='utf-8') as overall_f:
            overall_criteria_list = []
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in ['NRMS', 'NRMS-SA']:
                    with open('results/' + dataset + '/' + sub_dir + '/experiment_results-test.tsv', 'w', encoding='utf-8') as experiment_results_f:
                        experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                        criteria_list = []
                        for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                            if result_file[0] == '#' and result_file[-5:] == '-test':
                                with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                    line = result_f.read()
                                    if len(line.strip()) != 0:
                                        run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                        criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                        if len(criteria_list) > 0:
                            criteria_list.sort()
                            for criteria in criteria_list:
                                experiment_results_f.write(str(criteria) + '\n')
                            mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 = aggregate_criteria(criteria_list, experiment_results_f)
                            overall_criteria_list.append(OverallCriteria(sub_dir, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))
            overall_criteria_list.sort()
            for overall_criteria in overall_criteria_list:
                overall_f.write(str(overall_criteria) + '\n')


if __name__ == '__main__':
    aggregate_dev_result('MIND-small')
    aggregate_test_result('MIND-small')
    aggregate_dev_result('MIND-large')
    # For MIND-large, we submit prediction files to MIND leadboard for performance evaluation
