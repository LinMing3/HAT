from datasets import load_dataset,load_from_disk

# DATASET_ROOT = Path("/home/dangyunkai/yunkai/VLM/VIG-Group/jiacheng/251116-DynamicResolution/resolution_model/dataset/MME-train")
DATASET_ROOT = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"  

# dataset = load_dataset(
#     "json",
#     # path = DATASET_ROOT,
#     data_files=str(DATASET_ROOT / "MME_RealWorld.json"),
#     split="train[:10]",
# )
import numpy as np
train_dataset = load_from_disk(DATASET_ROOT)
print(train_dataset[0])
m_min, m_max = 0.0003634539953787473, 0.4404032400133952

print(np.argmax(np.array(train_dataset["mdf_raw"])))
print(train_dataset[5494])
e_min, e_max = 0.09866105056895785, 6.245104573220872
p_min, p_max = 257152.0, 1568000.0
print(m_min)
print(m_max)
print(e_min)
print(e_max)
print(p_min)    
print(p_max)

m1 = 0.004541
m2 = 0.002490
e1 = 5.359472
e2 = 5.690608
p1 = 1232*784
p2 = 1344*728.0

m_norm = (m1 - m_min) / (m_max - m_min + 1e-9)
e_norm = (e1 - e_min) / (e_max - e_min + 1e-9)
p_norm = (p1 - p_min) / (p_max - p_min + 1e-9)
print(m_norm, e_norm, p_norm)
print(np.mean([m_norm, e_norm, p_norm]))

m_norm = (m2 - m_min) / (m_max - m_min + 1e-9)
e_norm = (e2 - e_min) / (e_max - e_min + 1e-9)
p_norm = (p2 - p_min) / (p_max - p_min + 1e-9)
print(m_norm, e_norm, p_norm)
print(np.mean([m_norm, e_norm, p_norm]))

