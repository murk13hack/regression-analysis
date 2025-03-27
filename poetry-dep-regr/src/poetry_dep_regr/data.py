import numpy as np
import pandas as pd
from functools import cache

@cache
def exp_interp(start: float, end: float, num_points: int) -> list[float]:
    points = np.exp(np.linspace(np.log(start), np.log(end), num_points))[1:-1]
    return np.round(points, decimals=0).tolist()

data = {
    "in_queue": [
        9964, *exp_interp(7698, 9964, 6), 7698, *exp_interp(5419, 7698, 6), 5419,
        4857, 4428, 4429, 4180, 3384, 3118, 2911, 2864, 2830, 2818, 2799, 2748,
        2739, 2716, 2612, 2542, 2458, 2364, 2267, 2181, 2096, 1987, 1862
    ],
    "received": [
        1296, *exp_interp(652, 1296, 6), 652, *exp_interp(253, 652, 6), 253, 242,
        229, 227, 229, 151, 139, 140, 144, 147, 244, 181, 186, 153, 138, 135,
        129, 123, 99, 106, 96, 93, 94, 92
    ],
    "labels": [str(year) for year in range(1990, 2024)]
}

df = pd.DataFrame({
    "year": data["labels"],
    "in_queue": [x * 1000 for x in data["in_queue"]],
    "received": [x * 1000 for x in data["received"]]
})