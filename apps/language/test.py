import logging

import mteb

# from mteb.results import ModelResult
# from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
from waldo.models import build_model

model_config = {
    "model_name": "gemma3",
    "model_size": "1b",
    "device": "cuda:4",
}
print("ok")
model = build_model(config=model_config, load_offline=True, return_config=False)
print("done")

a = mteb
# b = ModelResult
# c = SentenceTransformer
