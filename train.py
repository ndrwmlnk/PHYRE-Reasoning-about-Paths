import argparse
from pathnet import PathSolver
import phyre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--setup", type=str, default="within")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()


    setup = "ball_"+args.setup+"_template"
    path = "final-test"

    solver = PathSolver(path, "pyramid", 64, fold=args.fold, setup=setup)
    print("Loading Training Data. Will be collected if not existing, may take a while.")
    solver.load_data(n_per_task=10)
    solver.train_supervised(epochs=args.epochs, train_mode = "COMB", setup=f'{args.setup}_{args.fold}')



    #train_ids, dev_ids, test_ids = phyre.get_fold(setup, args.fold)
    #train_ids = train_ids + dev_ids
    #dev_ids = tuple()
    #solver.load_data(n_per_task=1, shuffle=False, test=True, test_tasks=test_ids, setup_name="inspect4per"+("_cross" if setup == 'ball_cross_template' else ""), batch_size = 100)
    #solver.inspect_supervised(setup, args.fold, train_mode = "COMB", single_viz = False)