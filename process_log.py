import numpy as np


def process_log(filename, get_time = False, get_reward = False):
    with open(filename, "r") as f:
        lines = f.readlines()

    reward = []
    if get_reward:
        for line in lines:
            if line.startswith("INFO:root"):
                data = line.split(":")[-1]
                data = eval(data)
                for i, d in enumerate(data):
                    reward.append([i+1, d])
    run_time = []
    if get_time:
        for line in lines:
            if line.startswith("INFO:agent:frame"):
                tokens = line.split(",")
                # frame number
                frame = int(tokens[0].split(":")[-1].strip())
                # episode
                episode = int(tokens[1].split(":")[-1].strip())
                # wallclock time
                wallclock = float(tokens[-1].split(":")[-1].strip()) / 3600. # hours
                run_time.append([episode, wallclock, frame])
    
    return np.array(run_time), np.array(reward)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("process a log file")
    p.add_argument("filename", type=str)
    p.add_argument("--time", action="store_true")
    p.add_argument("--reward", action="store_true")

    args = p.parse_args()
    
    t, r = process_log(args.filename, args.time, args.reward)
    print(t)
    # print(r)