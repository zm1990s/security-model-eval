from flask import Flask, request, jsonify
import requests
import time

AZURE_API_URL = "https://xxx.cognitiveservices.azure.com/contentsafety/text:shieldPrompt?api-version=2024-09-01"
AZURE_API_KEY = "XXXX"

app = Flask(__name__)

def openai_to_azure(messages):
    user_prompt = None
    documents = []
    for msg in messages:
        if msg.get("role") == "user":
            if not user_prompt:
                user_prompt = msg.get("content")
            else:
                documents.append(msg.get("content"))
    return {
        "userPrompt": user_prompt or "",
        "documents": documents
    }

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    messages = data.get("messages", [])
    azure_payload = openai_to_azure(messages)
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    azure_response = requests.post(AZURE_API_URL, headers=headers, json=azure_payload)
    azure_json = azure_response.json()

    # 结果转换：OpenAI Compatible API 格式
    # 判断是否检测到攻击
    user_attack = azure_json.get("userPromptAnalysis", {}).get("attackDetected", False)
    documents_attack = [doc.get("attackDetected", False) for doc in azure_json.get("documentsAnalysis", [])]
    # 只要有一个为 True，则整体判定为攻击
    attack_detected = user_attack or any(documents_attack)

    # 取最后一个 user 消息内容作为返回
    user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            user_content = msg["content"]
            break

    # OpenAI Compatible API 响应
    response_content = "blocked" if attack_detected else user_content
    response = {
        "id": f"chatcmpl-azure-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "azure-content-safety",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_content.split()),
            "completion_tokens": len(response_content.split()),
            "total_tokens": len(user_content.split()) + len(response_content.split())
        },
        "azure_detection": {
            "user_attack": user_attack,
            "documents_attack": documents_attack,
            "attack_detected": attack_detected
        }
    }
    return jsonify(response), azure_response.status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)