import faiss
from pyspark.sql import SparkSession
import pandas as pd
from numpy.typing import NDArray
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

from lotus.dtype_extensions import convert_to_base_data
from lotus.models.faiss_rm import FaissRM


class IcebergRM(FaissRM):
    
    def __init__(self, 
                 spark: SparkSession,
                 max_batch_size: int = 64,
                 factory_string: str = "Flat",
                 metric=faiss.METRIC_INNER_PRODUCT,
   ):
        super().__init__(factory_string, metric)
        self.spark = spark
        self.max_batch_size = max_batch_size

    def _embed(self, docs: pd.Series | list) -> NDArray[np.float64]:
        
        table_name = docs.attrs["table_name"]
        column_name = docs.attrs["column_name"]
        all_embeddings = []
        
        embedding_json = json.loads(self.spark.sql(
            f"CALL local.system.load_table_embeddings( " +
             f"table => '{table_name}')").collect()[0]['embedding_json'])
        
        text_embeddings = dict()
        for all_embeddings_for_col in embedding_json:
            textEmbeddings = all_embeddings_for_col['textEmbeddings']
            for textEmbedding in textEmbeddings:
                if textEmbedding['textSegment']['metadata']['metadata']['column_name'] == column_name:
                    vecs = textEmbedding['embedding']['vector']
                    text = textEmbedding['textSegment']['text']
                    text_embeddings[text] = vecs
        for i in tqdm(range(0, len(docs), self.max_batch_size)):
            batch = docs[i : i + self.max_batch_size]
            _batch = convert_to_base_data(batch)
            embeddings = np.array([text_embeddings[text] for text in _batch])
            all_embeddings.append(embeddings)
 
        return np.vstack(all_embeddings)
    
    def __update_attrs__(self, docs: pd.Series | str | Image.Image | list | NDArray[np.float64], index_dir: str) -> None:
        """
        Update the attributes of the docs to include the table name and column name.
        This can be used later to get the embeddings from the iceberg table.
        """
        index_dir_array = index_dir.split(".")
        docs.attrs["table_name"] = index_dir_array[0] + '.' + index_dir_array[1]
        docs.attrs["column_name"] = index_dir_array[2]