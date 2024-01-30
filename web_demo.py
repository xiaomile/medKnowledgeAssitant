"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers.
We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""

from dataclasses import asdict

import json
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

from tools.transformers.interface import GenerationConfig, generate_interactive

logger = logging.get_logger(__name__)

from openxlab.model import download

download(model_repo='xiaomile/ChineseMedicalAssistant_internlm', output='xiaomile')

def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model(model_dir):
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.8, step=0.01)
        repetition_penalty=st.slider("repetition_penalty", 1.0, 1.1, 1.01, step=0.001)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)

    return generation_config


user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


def combine_history(prompt):
    messages = st.session_state.messages
    total_prompt = ""
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.replace("{user}", prompt)
    return total_prompt

# 由于不再使用json字符串格式，所以不再需要这两个输出转换函数
def convertMessage(cur_Message):
    try:
        new_message = json.loads(cur_Message)
        # print(new_message)
        if "question_type" in new_message.keys():
            if new_message["question_type"] == "smell":
                return new_message["name"]+"的气味是"+new_message["answer"]
            elif new_message["question_type"] == "alias":
                return new_message["name"]+"的别名是"+new_message["answer"]
            elif new_message["question_type"] == "part":
                return new_message["name"]+"所属的部是"+new_message["answer"]
            elif new_message["question_type"] == "cure":
                return new_message["name"]+"的功效是"+new_message["answer"]
            elif new_message["question_type"] == "symptom":
                return "治疗"+new_message["name"]+"的药方有"+new_message["answer"]
            else:
                return new_message["answer"]
        else:
            return new_message["answer"]
    except Exception as e:
        print(str(e))
        return cur_Message

# 由于不再使用json字符串格式，所以不再需要这两个输出转换函数
def preprocessMessage(message):
    if '"answer": "' in message:
        return message[message.find('"answer": "')+len('"answer": "'):].replace('"}','')
    else:
        return ""
    
def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    model, tokenizer = load_model('xiaomile')
    print("load model end.")

    user_avator = "doc/imgs/user.png"
    robot_avator = "doc/imgs/robot.png"

    st.title("中医药知识问答助手")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                # additional_eos_token_id=103028,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
