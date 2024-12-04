from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime, time, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pymongo.mongo_client import MongoClient
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from config import *
from transformers import AutoTokenizer
from gliner import GLiNER
from openai import OpenAI
from swarm import Swarm, Agent
from dateutil import parser
from zoneinfo import ZoneInfo
import os
os.environ['OPENAI_API_KEY'] = key

#manually trained bert model
model_trained = AutoModelForSequenceClassification.from_pretrained(f'{your_project_name}/improved_model')
tokenizer = AutoTokenizer.from_pretrained(f'{your_project_name}/improved_model')

#created flask app
app = Flask(__name__)

# Creating clients to use the serives
client = MongoClient(uri)
client_openai= OpenAI()
client_swarm = Swarm()

# Authenticating Google Calendar API
credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)

#translating agent
agent = Agent(
    name="Translator",
    instructions="Translate user query text to Gujarati.",
)
#mongodb collections
db = client['test_db']
customer_queries = db['customer_queries']
business_data = db['business_data']
calendar = db['calendar']
custom_responses = db['custom_responses']

model_entity = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

conversation_history=[]
#labels to identify from user input
labels = ["customer_name", "requested_date", "requested_time", "preferred_service"]


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/query', methods=['GET','POST'])
def handle_customer_query():
    query = request.form['query']

    # Tokenize the input sentence
    test_encoding = tokenizer(query, truncation=True, padding=True, return_tensors='pt')
    # Make a prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model_trained(**test_encoding)
        prediction = torch.argmax(outputs.logits, dim=-1)

    conversation = []
    assistant_message = 'kindly ask again'
    query_type=''
    while True:
        if prediction.item()==0:
            conversation.append({"role": "system", "content": f"You are a helpful AI receptionist of a tech company. Answer the users queries strictly from the context provided and example conversation which is a mongodb collection.\nContext: {business_data.find_one()}\nExample Conversation: {custom_responses.find_one({'query_type':'services_inquiry'})['response_template']}"})
            conversation.append({"role": "user", "content": 'User: '+query})
            query_type='services_inquiry'
            # Call API with updated conversation
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.2,
                max_tokens=150
            )
            assistant_message = (response.choices[0]).message.content

            #adding interactions into new collection
            customer_queries.insert_one({
                "query":query,
                "response":assistant_message
            })

        elif prediction.item()==1:
            if 'schedule' in query or 'book' in query:
                conversation.append({"role": "system", "content": f"You are a helpful AI receptionist of a tech company. Answer the users queries from strictly from the context provided and example conversation which is a mongodb collection.\nContext: {calendar.find_one()}\nExample Conversation: {custom_responses.find_one({'query_type':'scheduling_request'})['response_template']}"})
                conversation.append({"role": "user", "content": 'User: '+query})
                query_type='scheduling_request'
                # Call API with updated conversation
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation,
                    temperature=0.5,
                    max_tokens=150
                )
                assistant_message = (response.choices[0]).message.content
                customer_queries.insert_one({
                    "query":query,
                    "response":assistant_message
                })

        else:
            conversation.append({"role": "system", "content": f"You are a helpful AI receptionist of a tech company that provides software solutions. Answer the users queries strictly from the context provided and example conversation which is a mongodb collection. Also understand the query sentiment and respond sympathetically\nContext: {business_data.find_one()}\nExample Conversation: {custom_responses.find_one({'query_type':'other'})['response_template']}"})
            conversation.append({"role": "user", "content": 'User: '+query})
            query_type='other'
            # Call API with updated conversation
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.5,
                max_tokens=150
            )
            assistant_message = (response.choices[0]).message.content
            customer_queries.insert_one({
                "query":query,
                "response":assistant_message
            })
    
        # Append assistant's message to conversation history
        conversation.append({"role": "assistant", "content": assistant_message})

        #extracting entity with label
        entities = model_entity.predict_entities(query, labels)
        flag=0
        customer_name=''
        service=''
        time=''
        if entities:
            for entity in entities:
                if entity['label']=='requested_date':
                    date=entity['text']
                    flag=1
                if entity['label']=='customer_name':
                    customer_name=entity['text']
                if entity['label']=='requested_time':
                    time=entity['text']
                if entity['label']=='preferred_service':
                    service=entity['text']
        if flag:
            return redirect(url_for('schedule_appointment', customer_name=customer_name, requested_date=date, requested_time=time, preferred_service=service))
            # return redirect('confirm.html', customer_name=customer_name, requested_date=date, requested_time=time, preferred_service=service, query_type='scheduling_request')
        response_gujarati = client_swarm.run(agent=agent, messages=[{"role": "user", "content": assistant_message}])
        return render_template('home.html',resp=assistant_message, resp_g=response_gujarati.messages[-1]["content"], query_type=query_type)
    

@app.route('/schedule', methods=['GET','POST'])
def schedule_appointment():
    customer_name = request.args.get('customer_name')
    requested_date = request.args.get('requested_date')
    requested_time = request.args.get('requested_time')
    service = request.args.get('preferred_service')
  
    req_datetime= requested_date+' '+requested_time
    parsed_datetime = parser.parse(req_datetime, fuzzy=True)
    current_time = parsed_datetime.time()


    #Operating hours of company 10 to 6 so slot will be available between that
    min_time=time(10, 0)
    max_time=time(18,0)
    # Parse the requested datetime
    if min_time <= current_time <= max_time:
        # Set timezone (default to UTC)
        localized_datetime = parsed_datetime.replace(tzinfo=ZoneInfo('UTC'))
        end_datetime = localized_datetime + timedelta(hours=1)
        try:
            calendar_service = build('calendar', 'v3', credentials=credentials)
            start_datetime, end_datetime = localized_datetime.isoformat(), end_datetime.isoformat()
            
            # Prepare event details
            event = {
                'summary': f"Appointment with {customer_name}",
                'description': "Google meet",
                'start': {
                    'dateTime': start_datetime,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_datetime,
                    'timeZone': 'UTC',
                },
            }
            
            # Create the event
            event = calendar_service.events().insert(calendarId='primary', body=event).execute()
            message='Booking is confirmed'

            #updating calendar collection
            calendar.insert_one({
            "customer_name": customer_name,
            "service": service,
            "requested_date": requested_date,
            "requested_time": requested_time})
            return render_template('confirm.html', customer_name=customer_name, requested_time=requested_time, requested_date=requested_date, service=service, message=message)

        except Exception as e:
            message=(f"An error occurred in google calendar: {e}")

    else:
        message=f"Time {current_time} is outside the allowed slot of {min_time} to {max_time}. Please provide different time."          
        return render_template('confirm.html', msg=message)


@app.route('/admin', methods=['GET','POST'])
def admin_select():
    return render_template('admin_form.html')

@app.route('/admin_fill', methods=['GET','POST'])
def admin_fill():
    qt=request.form['query_type']
    temp=request.form['template']
    id=(custom_responses.find_one({"query_type":"services_inquiry"})['_id'])
    if id:
        custom_responses.update_one(
        {'_id': id},
        {'$set': {'field_name': temp}})
    else:
        custom_responses.insert_one({'query_type':qt,'response_template':temp})
    note='Template Updated'
    return render_template('admin_form.html', note=note)



if __name__ == '__main__':
    app.run(debug=True)
