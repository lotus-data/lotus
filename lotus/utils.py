import base64
import time
from io import BytesIO
from typing import Callable

import numpy as np
import pandas as pd
import requests  # type: ignore
from PIL import Image

import lotus


def cluster(col_name: str, ncentroids: int) -> Callable[[pd.DataFrame, int, bool], list[int]]:
    """
    Returns a function that clusters a DataFrame by a column using kmeans.

    Args:
        col_name (str): The column name to cluster by.
        ncentroids (int): The number of centroids to use.

    Returns:
        Callable: The function that clusters the DataFrame.
    """

    def ret(
        df: pd.DataFrame,
        niter: int = 20,
        verbose: bool = False,
        method: str = "kmeans",
    ) -> list[int]:
        import faiss

        """Cluster by column, and return a series in the dataframe with cluster-ids"""
        if col_name not in df.columns:
            raise ValueError(f"Column {col_name} not found in DataFrame")

        if ncentroids > len(df):
            raise ValueError(f"Number of centroids must be less than number of documents. {ncentroids} > {len(df)}")

        # get rmodel and index
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model using lotus.settings.configure()"
            )

        try:
            col_index_dir = df.attrs["index_dirs"][col_name]
        except KeyError:
            raise ValueError(f"Index directory for column {col_name} not found in DataFrame")

        if vs.index_dir != col_index_dir:
            vs.load_index(col_index_dir)
        assert vs.index_dir == col_index_dir

        ids = df.index.tolist()  # assumes df index hasn't been resest and corresponds to faiss index ids
        vec_set = vs.get_vectors_from_index(col_index_dir, ids)
        d = vec_set.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(vec_set)

        # get nearest centroid to each vector
        scores, indices = kmeans.index.search(vec_set, 1)

        # get the cluster centroids
        # centroids = kmeans.centroids
        # return indices.flatten(), scores.flatten(), centroids
        return indices.flatten()

    return ret


def fetch_image(image: str | np.ndarray | Image.Image | None, image_type: str = "Image") -> Image.Image | str | None:
    if image is None:
        return None

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, np.ndarray):
        image_obj = Image.fromarray(image.astype("uint8"))
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    elif image.startswith("s3://"):
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError

        try:
            import boto3

            s3 = boto3.client("s3")
            bucket_name, key = image[5:].split("/", 1)  # Split after "s3://"
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response["Body"].read()
            image_obj = Image.open(BytesIO(image_data))
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ValueError("AWS credentials not found or incomplete.") from e
        except Exception as e:
            raise ValueError(f"Failed to fetch image from S3: {e}") from e
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64, S3, and PIL.Image, got {image}"
        )
    image_obj = image_obj.convert("RGB")
    if image_type == "base64":
        buffered = BytesIO()
        image_obj.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return image_obj


def show_safe_mode(estimated_cost, estimated_LM_calls):
    print(f"Estimated cost: {estimated_cost} tokens")
    print(f"Estimated LM calls: {estimated_LM_calls}")
    try:
        for i in range(5, 0, -1):
            print(f"Proceeding execution in {i} seconds... Press CTRL+C to cancel", end="\r")
            time.sleep(1)
            print(" " * 60, end="\r")
        print("\n")
    except KeyboardInterrupt:
        print("\nExecution cancelled by user")
        exit(0)
