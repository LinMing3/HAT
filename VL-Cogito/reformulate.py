from datasets import load_from_disk
ds=load_from_disk("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/VL-Cogito/dataset_filtered_qwen25vl7b_50acc")
print(ds)
print(ds[0])
print(len(ds))

import numpy as np
buckets = np.array(ds['bucket'])

total = len(buckets)
for bucket_id in range(0,5):
    count = np.sum(buckets == bucket_id)
    pct = count / total * 100
    print(f"Bucket {bucket_id}: {count} samples, {pct:.2f}%")