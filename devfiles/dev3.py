import time
from tqdm import tqdm


for _ in tqdm(range(100), leave=False):
    time.sleep(0.1)

print('banana')