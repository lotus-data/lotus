Multimodal Models
===================

Overview
---------
Multimodal models combine textual and visual data to perform advanced tasks such as
image captioning, visual questions, and more. The ImageArray class enables handling of 
image data within a pandas DataFrame. Currently supports these image formats:
PIL images, numpy arrays, base64 strings, and image URLs

Initializing ImageArray
-----------------------
The ImageArray class is an extension array designed to handle images as data types in pandas. 
You can initilize an ImageArray with a list of supported image formats

.. code-block:: python

    from PIL import Image
    import numpy as np
    from lotus.utils import ImageArray

    # Example image inputs
    image1 = Image.open("path_to_image1.jpg")
    image2 = np.random.randint(0, 255, (100, 100, 3), dtype="uint8")

    # Create an ImageArray
    images = ImageArray([image1, image2, None])


Loading ImageArray
-------------------

The ImageArray supports multiple input formats for loading images.

- **PIL Images** : Directly pass a PIL image object
- **Numpy Arrays** : Convert numpy arrays to PIL Images automatically
- **Base64 Strings** : Decode base 64 strings into images
- **URLs** : Fetch images from HTTP/HTTPS URLs
- **File Paths** : Load images from local or remote file Paths
- **S3 URLs** : Fetch images stored in S3 buckets

Example:

.. code-block:: python

    from lotus.utils import fetch_image
    from PIL import Image

    image_path = "path_to_image.jpg"
    image_url = "https://example.com/image.png"
    base64_image = "data:image/png;base64,..." 

    # Load images
    pil_image = fetch_image(image_path)
    url_image = fetch_image(image_url)
    base64_image_obj = fetch_image(base64_image)
