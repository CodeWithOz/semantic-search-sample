import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, PodSpec
from pinecone_datasets import Dataset
from dotenv import load_dotenv
import datetime

from datasets import utils as datasets_utils


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")


# initialize pinecone
pc = Pinecone(api_key=pinecone_api_key)
spec = PodSpec(environment=pinecone_environment)
index_name = "semantic-search-fast"


# initialize index
if index_name not in pc.list_indexes().names():
    timed_print(f"Creating index {index_name!r}")
    pc.create_index(
        name=index_name,
        spec=spec,
        dimension=384,  # dimensionality of minilm
        metric="dotproduct",
    )
    timed_print(f"Created index {index_name!r}")

index = pc.Index(index_name)
timed_print(f"Index stats: {index.describe_index_stats()}")

if index.describe_index_stats().get("total_vector_count", 0) == 0:
    timed_print(f"Loading quora dataset subset")
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "datasets", "quora-bm25-dataset-80k-subset"
    )
    dataset = Dataset.from_path(file_path)
    timed_print(f"Loaded quora dataset subset")
    timed_print(f"Populating index with dataset")
    datasets_utils.upsert_dataset_redundantly(dataset, index, 100)
    timed_print(f"Populated index with dataset: {index.describe_index_stats()}")
else:
    timed_print(
        f"Index {index!r} contains {index.describe_index_stats().get('total_vector_count', 0)} vectors"
    )

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
query = "what countries are favorable to digital nomads?"

# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(vector=xq, top_k=5, include_metadata=True)

timed_print(f"Original query: {query}")
for match in xc["matches"]:
    timed_print(f"Match score: {match['score']}; match text: {match['metadata']['text']}")
