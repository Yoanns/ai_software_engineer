# app.py
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, session
from models import SoftwareEngineerAgent, create_model
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-123')

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

@app.route('/solve', methods=['POST'])
def solve():
    try:
        # if 'agent' not in session:
        #     logger.warning("Solve attempt without configured agent")
        #     return render_template('error.html', error="Agent not configured")

        model = create_model(session['model'])
        agent = SoftwareEngineerAgent(model)
        agent.__dict__ = session['agent']

        problem = request.form['problem']
        logger.info(f"Processing problem: {problem[:50]}...")  # Log first 50 chars

        result = agent.process_task(problem)
        session['agent'] = agent.__dict__

        logger.info(f"Problem processed successfully: {result['success']}")
        return render_template('results.html', result=result)

    except Exception as e:
        logger.error(f"Error processing solution: {str(e)}", exc_info=True)
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)