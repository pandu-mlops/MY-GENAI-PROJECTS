
import streamlit as st
import requests

# Flask backend URL
BACKEND_URL = "http://backend:5000/chat"

# Page configuration
st.set_page_config(
    page_title="GenAI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Application title
st.title("🤖 My GenAI Chatbot")

st.subheader("Powered by Gemma 3 1B and Ollama")

st.write("Ask me anything!")

st.divider()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Enter your question...")

if question:

    # Save current question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    # Prepare previous conversation
    previous_messages = st.session_state.messages[:-1]

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Gemma 3 1B is thinking..."):

            try:

                # Send request to Flask backend
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "question": question,
                        "messages": previous_messages
                    },
                    timeout=600
                )

                # Check API response
                response.raise_for_status()

                result = response.json()

                # Extract generated answer
                answer = result.get("answer", "")

                if not answer:
                    answer = "Sorry, the model returned an empty response."

                # Display answer
                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.exceptions.Timeout:

                st.error(
                    "Request timed out. Please try again."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to Flask backend. "
                    "Please check Docker containers."
                )

            except requests.exceptions.HTTPError as e:

                st.error(f"Flask API Error: {e}")

            except Exception as e:

                st.error(f"Unexpected Error: {e}")


# Clear conversation button
if st.button("Clear Conversation"):

    st.session_state.messages = []

    st.rerun()


# Footer
st.divider()

st.caption(
    "GenAI Chatbot | Streamlit + Flask + Ollama + Gemma 3 1B"
)
