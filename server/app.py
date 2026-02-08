import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from services.buffer_service import BufferService
from services.chat_service import ChatService

load_dotenv()
app = Flask(__name__)
CORS(app)

buffer_service = BufferService()
chat_service = ChatService()


@app.route('/buffer/record', methods=['POST'])
def start_context_buffer():
    return jsonify(buffer_service.start())


@app.route('/buffer/kill', methods=['POST'])
def kill_buffer():
    return jsonify(buffer_service.stop())


@app.route('/buffer/status', methods=['GET'])
def get_status():
    return jsonify(buffer_service.get_status())


def sse_stream(generator):
    def generate():
        for chunk in generator:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route('/assist', methods=['POST'])
def assist_request():
    snapshots = buffer_service.flush_buffer(clear=True)
    return sse_stream(chat_service.init_chat_stream(snapshots))


@app.route('/assist/chat', methods=['POST'])
def chat():
    message = request.json.get("message")
    return sse_stream(chat_service.send_message_stream(message))


@app.route('/')
def default():
    print("hello world")
