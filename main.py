import json
import re
import streamlit as st
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr
import threading
import folium
from streamlit_folium import st_folium
from geopy.distance import great_circle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from streamlit_geolocation import streamlit_geolocation

# Assuming jsondat is the filename (with .json extension)
filename = "test.json"

# Open the file and load the data
with open(filename, 'r') as file:
    data = json.load(file)

# Now you can access the email
email = data["email"]
print(email)

# Function to fetch and verify location
def fetch_location():
    location = streamlit_geolocation()
    print(location)
    if location:
        latitude = location['latitude']
        longitude = location['longitude']
        # Ensure latitude and longitude are greater than 0
        if latitude==None and longitude==None:
            st.error("Kindly click the icon above and accept live location")
            st.stop()
        else:
            return latitude, longitude

    else:
        st.error("Location retrieval failed.")
        st.stop()

# Load environment variables
load_dotenv()

# Set up Google Generative AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "AIzaSyADAMRi3OSq5CkfT1uESjk4EKbU47ybBTE"

def get_day_of_week():
    # Get the current date
    current_date = datetime.datetime.now()
    # Get the day of the week as a string
    day_of_week = current_date.strftime("%A")
    return day_of_week
print(get_day_of_week())
def send_email(subject, body, to_email):
    # Gmail SMTP server configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username ='davidvictor295@gmail.com'
    # smtp_username = os.getenv('GMAIL_USERNAME')
    smtp_password ="sjiv gwjx giig nrtq"
    # smtp_password = os.getenv('GMAIL_PASSWORD')

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Function to calculate distance and estimated travel time
def calculate_distance_and_time(loc1, loc2, speed_kmh=50):
    distance_km = great_circle(loc1, loc2).kilometers
    time_hours = distance_km / speed_kmh
    return distance_km, time_hours

# def get_location():
#     g = geocoder.ip('me')
#     return g.latlng

# # Get location
# location = get_location()
# if location:
#     lat, lon = location
# Function to display the map
def display_map(latitude, longitude):
    # Example coordinates for the ambulance and request locations
    ambulance_location = (8.72027334912902, 4.480440093756269)  # Replace with real coordinates
    request_location = (latitude, longitude)    # Replace with real coordinates

    # Calculate distance and estimated travel time
    distance_km, time_hours = calculate_distance_and_time(ambulance_location, request_location)

    # Create a Folium map centered around the midpoint
    map_center = ((ambulance_location[0] + request_location[0]) / 2, (ambulance_location[1] + request_location[1]) / 2)
    m = folium.Map(location=map_center, zoom_start=12)

    # Add markers for ambulance and request locations
    folium.Marker(location=ambulance_location, popup="Ambulance Location", icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker(location=request_location, popup="Request Location", icon=folium.Icon(color='red')).add_to(m)

    print(latitude, longitude)

    return m, distance_km, time_hours


# Function to extract text from the specified JSON file
def get_json_text(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    staff_descriptions = data.get('staff_descriptions', [])
    text = "\n".join(staff_descriptions)
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question, make sure to provide Which doctor and driver that is on duty, give full details about them and phone number dont give empty reply\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get Gemini's response
def get_gemini_response(user_question):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.predict(user_question)
    return response

# Function to search json for staffs based on the user input
def search_json_for_staffs(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_query)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    return response

# Function to convert text to speech asynchronously
def text_to_speech_async(text):
    def run_speech_engine(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=run_speech_engine, args=(text,))
    thread.start()

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.write("You said: ", text)
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            st.write("Could not request results; check your network connection.")
            return ""

# Function to process user input and generate a response
def user_input(user_question):
    gemini_response = get_gemini_response(user_question)
    st.session_state["gemini_response"] = gemini_response
    st.session_state["show_recommendation"] = True
    text_to_speech_async(gemini_response)

# Function to authenticate admin login
def authenticate(username, password):
    return username == "admin" and password == "12345"

# Function to load and edit JSON
def edit_json():
    json_data = {}

    try:
        with open("test.json", "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        st.error("JSON file not found. Please make sure 'test.json' exists in the current directory.")
        return

    st.header("Edit JSON")
    st.write("Current JSON Content:")
    st.write(json_data)

    new_content = st.text_area("Edit JSON content:", value=json.dumps(json_data, indent=4))

    if st.button("Save JSON"):
        try:
            updated_data = json.loads(new_content)
            with open("test.json", "w") as f:
                json.dump(updated_data, f, indent=4)
            st.success("JSON updated successfully!")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
        except Exception as e:
            st.error(f"Error saving JSON: {e}")

# Function to get the doctor and driver on duty
def get_duty_staff(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return response["output_text"]

def main():
    # Fetch the location at the start
    latitude, longitude = fetch_location()
    
    tabs = ["Emergency", "Interaction", "Admin"]
    tab_choice = st.sidebar.radio("Navigation", tabs)

    if tab_choice == "Emergency":
        st.header("Kwasu Emergency Dispatch System 🚑")
        
        map_obj, distance_km, time_hours = display_map(latitude, longitude)  # Pass latitude and longitude to display_map
        st_folium(map_obj, width=700, height=500)  # Display the map
        st.write(f"Distance between you and the emergency location is {distance_km:.2f} km")
        st.write(f"Estimated travel time is approximately {time_hours:.2f} hours at 50 km/h")   
        if st.button("Request Ambulance"):
            duty_query = f"today is {get_day_of_week()} i need doctor and a driver on duty"
            duty_response = get_duty_staff(duty_query)
            st.write(duty_response)
            pattern = r'0\d{10}'
            matches = re.findall(pattern, duty_response)
            if len(matches) >= 2:
                st.write(matches[0], matches[1])
                st.markdown(f"Doctor:  [📞 {matches[0]}](tel:{matches[0]}) Driver:  [📞 {matches[1]}](tel:{matches[1]})")
                st.markdown(f"[chat with doctor](https://wa.me/+234{matches[0]}) [chat with driver](https://wa.me/+234{matches[1]})")

            else:
                st.write("Not enough phone numbers found")
            # Send the email
            subject = "Ambulance Request"
            body = f"Request Location: https://www.google.com/maps/?q={latitude},{longitude}\n\nDetails:\n{duty_response}"
            to_email = "mustaphay456@gmail.com"  # Replace with the actual email address

            if send_email(subject, body, to_email):
                st.success("Ambulance request sent successfully!")
            else:
                st.error("Failed to send ambulance request.")

    elif tab_choice == "Interaction":
        st.header("Ask me any question about your health 💁")
        json_path = "test.json"
        raw_text = get_json_text(json_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database Indexed Successfully.")

        user_question = st.text_input("Ask a Question", key="user_question_text")
        if st.button("Speak a Question"):
            user_question = speech_to_text()
            if user_question:
                user_input(user_question)
        elif st.button("Submit", key="submit_question"):
            if user_question:
                user_input(user_question)
        
        if "gemini_response" in st.session_state:
            st.write("JoanAI's Response: ", st.session_state["gemini_response"])

            if st.session_state.get("show_recommendation", False):
                recommend = st.radio("Do you want to request an emergency?", ("Yes", "No"), key="recommend")
                if recommend == "Yes":
                    if st.button("Request Ambulance 🚑💉💊😷🧑‍⚕️"):
                        duty_query = f"today is {get_day_of_week()} i need doctor and a driver on duty"
                        duty_response = get_duty_staff(duty_query)
                        pattern = r'0\d{10}'
                        
                        matches = re.findall(pattern, duty_response)
                        if len(matches) >= 2:
                            st.write(matches[0], matches[1])
                        else:
                            st.write("Not enough phone numbers found")


    elif tab_choice == "Admin":
        st.header("Admin Section")

        if "admin_logged_in" not in st.session_state:
            st.subheader("Login to Admin Panel")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate(username, password):
                    st.session_state["admin_logged_in"] = True
                    st.success(f"Logged in as {username}")
                else:
                    st.error("Invalid username or password")
        else:
            edit_json()

    if st.sidebar.button("Clear Cache"):
        st.sidebar.success("Cache Cleared")

if __name__ == "__main__":
    main()
