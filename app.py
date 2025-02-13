# app.py
import logging
from datetime import timedelta
import time
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, session, Response
from models import SoftwareEngineerAgent, create_model
import os
from dotenv import load_dotenv
import json
import uuid


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-123')
app.permanent_session_lifetime = timedelta(minutes=30)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=1024*1024*5,  # 5MB
            backupCount=10
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a global agent store (simple in-memory for demonstration)
current_agents = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            model_type = request.form['model_type']
            try:
                model = create_model(model_type)
                session['model'] = model_type
                session['agent'] = SoftwareEngineerAgent(model).__dict__
                logger.info(f"Model configured: {model_type}")
            except Exception as e:
                logger.error(f"Model configuration failed: {str(e)}", exc_info=True)
                return render_template('error.html', error=str(e))

        return render_template('index.html')
    except Exception as e:
        logger.critical(f"Unexpected error in index route: {str(e)}", exc_info=True)
        return render_template('error.html', error="Internal server error")

@app.route('/progress')
def progress():
    # Capture the session id in the request context
    session_id = session.get("session_id")
    if not session_id:
        return "Session ID not set", 400

    def generate():
        agent = current_agents.get(session_id)
        while agent:
            progress_data = agent.get_progress()
            yield f"data: {json.dumps(progress_data)}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/configure-model', methods=['POST'])
def configure_model():
    try:
        session.permanent = True
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

        model_type = request.form['model_type']
        model = create_model(model_type)

        # Store agent in memory with session association
        current_agents[session_id] = SoftwareEngineerAgent(model)

        session['model'] = model_type
        logger.info(f"Model configured: {model_type}")
        return '', 204
    except Exception as e:
        logger.error(f"Model configuration failed: {str(e)}", exc_info=True)
        return str(e), 400

@app.route('/solve', methods=['POST'])
def solve():
    try:
        if 'model' not in session:
            logger.warning("Solve attempt without configured model")
            return render_template('error.html', error="Model not configured")

        # Create fresh agent for each request
        model = create_model(session['model'])
        agent = SoftwareEngineerAgent(model)

        problem = request.form['problem']
        logger.info(f"Processing problem: {problem[:50]}...")

        result = agent.process_task(problem)

        logger.info(f"Problem processed successfully: {result['success']}")
        return Response(render_template('results.html', result=result))

    except Exception as e:
        logger.error(f"Error processing solution: {str(e)}", exc_info=True)
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)