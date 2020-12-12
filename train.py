import argparse
from pathnet import PathSolver
import phyre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="fold number of the test tasks to evaluate on")
    parser.add_argument("--setup", type=str, default="within", help="evaluations etup: 'cross' or 'within'")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs to train")
    parser.add_argument("--samples-per-task", type=int, default=10, help="number of samples per task for training, higher number will increase the time for train data set colelction")
    args = parser.parse_args()


    setup = "ball_"+args.setup+"_template"
    path = "final-test"

    solver = PathSolver(path, "pyramid", 64, fold=args.fold, setup=setup)
    print("Loading Training Data. Will be collected if not existing, may take a while.")
    solver.load_data(n_per_task=args.samples_per_task)
    solver.train_supervised(epochs=args.epochs, train_mode = "COMB", setup=f'{args.setup}_{args.fold}')