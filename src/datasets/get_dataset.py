import os
import datetime
from dotenv import load_dotenv
from pinecone_datasets import load_dataset


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


timed_print(f"loading quora dataset")
dataset = load_dataset("quora_all-MiniLM-L6-bm25")
timed_print(f"loaded quora dataset")

timed_print(f"dropping metadata column")
dataset.documents.drop(["metadata"], axis=1, inplace=True)
timed_print(f"dropped metadata column")

timed_print(f"renaming blob to metadata column")
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)
timed_print(f"renamed blob to metadata column")

# we will use 80K rows of the dataset between rows 240K -> 320K
timed_print(f"dropping unused rows")
dataset.documents.drop(dataset.documents.index[320_000:], inplace=True)
dataset.documents.drop(dataset.documents.index[:240_000], inplace=True)
timed_print(f"dropped unused rows")

# save the dataset locally
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quora-bm25-dataset-80k-subset")
timed_print(f"saving dataset locally")
dataset.to_path(file_path)
timed_print(f"saved dataset locally")
