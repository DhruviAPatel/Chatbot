import streamlit as st
import backend as demo
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Document Q&A Chatbot")

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Document Q&A Chatbot 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)

# Multiple file uploader
uploaded_files = st.file_uploader("Upload files", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        try:
            file_paths = []
            for uploaded_file in uploaded_files:
                with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    file_paths.append(tmp_file.name)
                    st.write(f"File saved at: {tmp_file.name}")
            
            # Debugging print statement
            st.write("Files uploaded:", file_paths)
            
            if not file_paths:
                st.error("No files uploaded.")
            else:
                # Debugging print statement
                st.write("Processing files:", file_paths)
                try:
                    st.session_state.vector_index = demo.process_files(file_paths)
                    st.success("Files processed and indexed successfully.")
                except ValueError as e:
                    st.error(f"Processing error: {e}")
                    st.write("Files that caused the error:", file_paths)
        except Exception as e:
            st.error(f"Error processing uploaded files: {e}")

input_text = st.text_area("Ask a question about the uploaded documents", label_visibility="collapsed")
go_button = st.button("Ask")

if go_button and input_text:
    with st.spinner("Finding the answer..."):
        try:
            if 'vector_index' not in st.session_state:
                st.error("Please upload and process documents first.")
            else:
                response_content = demo.pdeu_rag_response(index=st.session_state.vector_index, question=input_text)
                st.write(response_content)
        except Exception as e:
            st.error(f"Error generating response: {e}")
