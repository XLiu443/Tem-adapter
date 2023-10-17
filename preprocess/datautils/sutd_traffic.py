import json
from datautils import utils
import nltk
from collections import Counter
import torch
import pickle
import numpy as np
import jsonlines
import pandas as pd
import os


def load_video_paths(args):
    """Load a list of (path,video_id tuples)."""
    input_paths = []
    item_list = []
    first = True
    with open(args.video_file, "r") as f:
        for item in jsonlines.Reader(f):
            if first:
                cols = item
                first = False
            else:
                if not (type(item[1]))==int:
                    continue
                item_list.append(item)

    csv_data = pd.DataFrame(item_list, columns=cols)

    video_names = list(csv_data["vid_filename"])
    video_ids = list(csv_data["vid_id"])

    for idx, video in enumerate(video_names):
        video_abs_path = os.path.join(args.video_dir, video)
        input_paths.append((video_abs_path, video_ids[idx]))
    input_paths = list(set(input_paths))

    return input_paths


