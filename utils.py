def vpnan():
    for i in range(5):
        f = open(f"vp/{i}.txt", "r")
        out = open(f"vp/out/{i}.txt", "w")
        lines = f.readlines()
        f = open(f"vp/{i}.txt", "w")
        k = 0
        for l in lines:
            if k != 0 and l.strip("\n") != "nan nan":
                f.write(l)
            elif k != 0:
                out.write(str(k)+"\n")
            k += 1

def concfiles():
    filenames = [f'vp/{i}.txt' for i in range(5)]
    with open('vp/vps.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def delnanframes():
    import os
    import numpy as np
    directory = os.fsencode("_optflow")
    reject = np.loadtxt("vp/reject.txt")
    for file in os.listdir(directory):
        nbframe = int(os.fsdecode(file).split(".")[0])
        if nbframe in reject:
            os.remove(f"_optflow/{nbframe}.png")

if __name__ == "__main__":
    delnanframes()

