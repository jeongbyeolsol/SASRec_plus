import json
from pathlib import Path

# 현재 스크립트 파일이 위치한 디렉토리 경로
script_dir = Path(__file__).parent 

# JSON 파일명
json_filename = 'MicroLens-100k_user_sequences.json'

# 파일의 전체 경로를 Path 객체로 구성
json_path = script_dir / json_filename

with open(json_path, "r") as f:
    data = json.load(f)  # dict: user_id -> [item1, item2, ...]

# ---- 1) 유저 수 ----
num_users = len(data)

# ---- 2) 유니크 아이템 수 ----
item_set = set()
for seq in data.values():
    item_set.update(seq)

min_user_id = 99

for seq in data.keys():
  if min_user_id > int(seq):
    min_user_id = int(seq)

num_items = max(item_set)  # PAD=0 고려하면 embedding size는 이렇게 잡는 것이 일반적

print("가장 작은 유저 index:", min_user_id)
print("유저 수 =", num_users)
print("아이템 수 =", len(item_set))
print("embedding용 num_items =", num_items)
