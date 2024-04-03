import pandas as pd
import sys
import random

def main():
    #must be absolute
    target = sys.argv[1]
    data = pd.DataFrame(pd.read_pickle(target))
    
    print("RANDOM ROW:")
    print(data.iloc[random.randint(0, len(data))])

if __name__ == "__main__":
    main()

