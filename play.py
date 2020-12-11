import argparse
from pathnet import PathSolver
import phyre
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--setup", type=str, default="within")
    parser.add_argument("-eval", action="store_true")
    args = parser.parse_args()



    setup = "ball_"+args.setup+"_template"
    path = "final-test"

    if not args.templates:
        train_ids, dev_ids, test_ids = phyre.get_fold(setup, args.fold)
        tasks = test_ids
    else:
        tasks = args.templates.split(",")

    solver = PathSolver(path, "pyramid", 64, fold=args.fold, setup=setup)
    solver.load_models(load_from=-1)
    actions, pipelines = solver.get_actions(tasks)

    if args.eval:
        print()
        os.makedirs("./solutions/", exist_ok = True)
        sim = phyre.initialize_simulator(tasks, 'ball')
        eva = phyre.Evaluator(tasks)
        for task_idx, task in enumerate(tasks):
            print("at task", task)
            solved = False
            for actionidx, action in enumerate(actions[task_idx]):
                res = sim.simulate_action(task_idx, action,  need_featurized_objects=False)
                eva.maybe_log_attempt(task_idx, res.status)
                if not res.status.is_invalid() and not solved and eva.attempts_per_task_index[task_idx]<=100:
                    line = (np.array(pipelines[task_idx]))
                    img = np.zeros((5,5,66,66,3))
                    for fridx, frame in enumerate(res.images):
                        obs = phyre.observations_to_uint8_rgb(frame)
                        obs = cv2.resize(obs, (64,64))
                        obs = np.pad(obs, ((1,1),(1,1),(0,0)))
                        img[int(fridx/5), fridx%5] = obs
                    #print(img.shape)
                    img = np.concatenate(img, axis=1)
                    #print(img.shape)
                    img = np.concatenate(img, axis=1)
                    #print(img.shape)
                    img = np.concatenate((line, img), axis=0)
                    img = img.astype(np.uint8)
                    #print(img.shape)
                    #print(np.max(img), np.max(line))
                    os.makedirs(f"./solutions/{task}/", exist_ok = True)
                    plt.imsave(f"./solutions/{task}/attempt{actionidx}.png",img)

                if eva.attempts_per_task_index[task_idx]>=100:
                    break

                if res.status.is_solved():
                    solved = True
        print(eva.compute_all_metrics())

    print(actions[0])