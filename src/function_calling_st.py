import streamlit as st
import json
import requests
# from ---- для модели

tools = [
            {
                "name": "sell_item",
                "description": "Продать предмет игроку",
                "parameters": { 
                    "type": "object", 
                    "properties": { 
                    "item_name": { 
                        "type": "string", 
                        "description": "Название предмета" 
                    }, 
                    "price": { 
                        "type": "number",
                        "description": "Цена в монетах" 
                        }
                    }, 
                    "required": [ "item_name", "price" ] 
                }
            },
            {
                "name": "buy_item",
                "description": "Купить предмет у игрока",
                "parameters": { 
                    "type": "object",
                    "properties": { 
                        "item_name": { 
                            "type": "string", 
                            "description": "Название предмета"
                            }, 
                        "price": { 
                            "type": "number", 
                            "description": "Цена в монетах" 
                        } 
                    }, 
                    "required": [ "item_name", "price" ] 
                }
            },
            {
                "name": "get_quests",
                "description": "Получить квесты",
                "parameters": { 
                    "type": "object",
                    "properties": {
                        "difficulty": { 
                            "type": "string", 
                            "description": "Сложность квестов", 
                            "enum": ["easy", "medium", "hard"] 
                        },
                        "description": {
                            "easy" : ["Find a farmer's cat", "Build a tree house for kids"],
                            "medium" : ["Catch an evil robber in town"],
                            "hard" : ["Kill a dragon"]
                        }
                    },
                    "required": [ "difficulty" ] 
                }
            },

            {
                "name": "get_news",
                "description": "Получить новости",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "description": "Список тем",
                            "items": {
                                "type": "string",
                                "enum": [ "война", "королевство", "урожай" ] 
                            }
                        }
                    },
                    "required": [ "themes" ] 
                }
            }
        ]


def buy_items(items, prices):
    """Adventurer can buy items for his inventory"""
    list_of_items = {
        "items" : items,
        "prices" : prices
    }
    return json.dump(list_of_items)


def get_quests(level, easy, medium, hard):
    """Adventurer is given list of available quests"""
    list_of_quests = {
        "level" : level,
        "easy" : easy,
        "medium" : medium,
        "hard" : hard
    }
    return json.dump(list_of_quests)

def order_beer(count, price = 10):
    """Order beer and pay by coins"""
    order = {
        "count" : count,
        "price" : price
    }
    return json.dump(order)

def tell_news():
    pass


def model_response(model, prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = model.generate(prompt_input)
    return output

def show_main_page():
    # st.session_state.keep_graphics = True
    inventory = {}
    money = {}

    st.title("Добро пожаловать в таверну!")
    st.image("d81ae2_99c9dfa697944a0b9201348b7b60c875~mv2.jpg")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Чем сегодня могу Вам помочь, авантюрист?"}]

    for message in st.session_state.messages:
        if message['role'] == "assistant":
            with st.chat_message(message["role"], avatar = "bartender-drink-occupation-avatar-512.jpg"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar = "man-avatar-boy-adventure-fashion-1024.jpg"):
                st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar = "man-avatar-boy-adventure-fashion-1024.jpg"):
            st.write(prompt)

    

# Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar = "bartender-drink-occupation-avatar-512.jpg"):
            with st.spinner("Хмммм..."):
                # response = model_response(model, prompt) ---- надо поменять
                response = ["this should be model response"]
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if __name__ == "__main__":
    show_main_page()
