Fast Data Loading
=================

Geti Library provides several ways to boost model training speed,
one of which is fast data loading.


=======
Caching
=======


*****************
In-Memory Caching
*****************
Geti Library provides in-memory caching for decoded images in main memory.
If the batch size is large, such as for classification tasks, or if dataset contains
high-resolution images, image decoding can account for a non-negligible overhead
in data pre-processing.
One can enable in-memory caching for maximizing GPU utilization and reducing model
training time in those cases.


.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from getitune.data.module import DataModule

         datamodule = DataModule(..., mem_cache_size="8GB")

   .. tab-item:: CLI

      .. code-block:: shell

         (getitune) ...$ getitune train ... --data.mem_cache_size 8GB
