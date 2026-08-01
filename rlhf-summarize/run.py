
from train_sft import train_sft

import sys
from train_sft import train_sft
from train_rm import train_rm


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify one of: sft, rm, ppo, all")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == "sft":
        train_sft()
    elif arg == "rm":
        train_rm()
    # elif arg == "ppo":
    #   train_ppo()
    elif arg == "all":
        train_sft()
        train_rm()
        # train_ppo()

    else:
        print("Unknown argument:", arg)
        print("Please specify one of: sft, rm, ppo, all")
        sys.exit(1)
