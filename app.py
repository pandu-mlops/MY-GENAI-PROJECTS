
import os
import requests

from flask import Flask, request, jsonify

app = Flask(__name__)

# Ollama configuration
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434/api/chat"
)

MODEL_NAME = "gemma3:1b"

# Request timeout
REQUEST_TIMEOUT = 600


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Welcome to GenAI Chatbot API",
        "status": "Running",
        "model": MODEL_NAME
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "Flask Backend"
    })


@app.route("/chat", methods=["POST"])
def chat():

    try:

        # Read JSON request
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Please send valid JSON data"
            }), 400

        # Get question
        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "error": "Question is required"
            }), 400

        # Read previous conversation messages
        messages = data.get("messages", [])

        # Validate conversation history
        if not isinstance(messages, list):
            return jsonify({
                "error": "Messages must be a list"
            }), 400

        # Build Ollama conversation
        chat_messages = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role in ["user", "assistant"] and isinstance(content, str):
                chat_messages.append({
                    "role": role,
                    "content": content
                })

        # Add current question
        chat_messages.append({
            "role": "user",
            "content": question
        })

        # Ollama request payload
        payload = {
            "model": MODEL_NAME,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "num_predict": 100
            }
        }

        print("Sending request to Ollama...")

        # Call Ollama
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        # Extract model response
        answer = result.get(
            "message", {}
        ).get("content", "")

        if not answer:
            return jsonify({
                "error": "Ollama returned an empty response"
            }), 502

        print("Gemma response generated successfully.")

        return jsonify({
            "question": question,
            "answer": answer,
            "model": MODEL_NAME
        }), 200

    except requests.exceptions.Timeout:

        return jsonify({
            "error": "Ollama request timed out. Please try again."
        }), 504

    except requests.exceptions.ConnectionError:

        return jsonify({
            "error": "Cannot connect to Ollama. Check the Ollama container."
        }), 503

    except requests.exceptions.RequestException as e:

        return jsonify({
            "error": f"Ollama API error: {str(e)}"
        }), 502

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
