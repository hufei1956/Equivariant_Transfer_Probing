import json
import os
import requests
from urllib.parse import urljoin

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'label', 'project-4-at-2026-01-16-15-58-932da740.json')
SAVE_DIR = os.path.join(BASE_DIR, 'aligned_images')
LS_URL = 'http://localhost:8080'             # 你的 Label Studio 地址
# 注意：通常 Label Studio 的 API Token 是 40 位十六进制字符串（在 Account & Settings -> Access Token 获取）
# 如果遇到 401/403 错误，请检查 Token 是否正确
API_TOKEN = '7c287f1ce7cb21b892d2d7f738509a0ed4a653e4'

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# 检查 JSON 文件是否存在
if not os.path.exists(JSON_PATH):
    print(f"错误: 找不到 JSON 文件: {JSON_PATH}")
    exit(1)

# 读取 JSON
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
except json.JSONDecodeError:
    print(f"错误: JSON 文件格式不正确: {JSON_PATH}")
    exit(1)

headers = {
    "Authorization": f"Token {API_TOKEN}"
}

print(f"开始下载 {len(tasks)} 张图片...")

success_count = 0
fail_count = 0

for i, task in enumerate(tasks):
    # 获取图片 URL (根据实际 JSON 结构调整键名)
    img_url_relative = task.get('data', {}).get('image') or task.get('data', {}).get('img')
    
    if not img_url_relative:
        print(f"[{i+1}/{len(tasks)}] 跳过: 未找到图片 URL (Task ID: {task.get('id')})")
        fail_count += 1
        continue

    # 拼接完整 URL
    full_url = urljoin(LS_URL, img_url_relative)
    
    # 提取文件名
    original_filename = os.path.basename(img_url_relative)
    save_path = os.path.join(SAVE_DIR, original_filename)

    # 下载
    if os.path.exists(save_path):
        print(f"[{i+1}/{len(tasks)}] 跳过: 文件已存在 {original_filename}")
        success_count += 1
        continue
        
    try:
        response = requests.get(full_url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[{i+1}/{len(tasks)}] 下载成功: {original_filename}")
            success_count += 1
        else:
            print(f"[{i+1}/{len(tasks)}] 下载失败 (Status {response.status_code}): {full_url}")
            if response.status_code in [401, 403]:
                print("  提示: 请检查 API_TOKEN 是否正确")
            fail_count += 1
    except Exception as e:
        print(f"[{i+1}/{len(tasks)}] 下载出错: {e}")
        fail_count += 1

print("-" * 30)
print(f"下载完成! 成功: {success_count}, 失败: {fail_count}")
print(f"图片保存至: {SAVE_DIR}")