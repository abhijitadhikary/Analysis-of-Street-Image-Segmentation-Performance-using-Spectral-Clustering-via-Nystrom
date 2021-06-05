from evaluation_metrics import run_evaluation

if __name__ == '__main__':
    run_evaluation('train')
    run_evaluation('val')
    run_evaluation('test')