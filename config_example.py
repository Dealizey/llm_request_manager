proxy = "http://127.0.0.1:xxx"
# proxy = None

# OhMyGPT
api_key = "sk-"
api_base = "https://api.ohmygpt.com/v1"

# 硅基流动
silicon_flow_api_base = "https://api.siliconflow.cn/v1"
silicon_flow_api_key = "sk-"

# 阿里云
ali_api_key = "sk-"
ali_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

model_to_use = "gemini-2.0-flash-001"

api_base = api_base.rstrip("/")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}
if proxy:
    proxies = {"http": proxy, "https": proxy}
else:
    proxies = {}
    