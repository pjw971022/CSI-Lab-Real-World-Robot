import os
import google.generativeai as genai


os.environ['GOOGLE_API_KEY'] = 'AIzaSyDRv4MkxqaTD9Nn4xDieqFkHbf8Ny4eU_I'
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# Select the model
model = genai.GenerativeModel('gemini-pro')
text_config = {"stop_sequences" : ['# Query: ','objects = '],"max_output_tokens": 512,  "top_p": 1 , "temperature": 0} #  
safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },]

file_path = '/home/andykim0723/RLBench/VoxPoser/src/prompts/rlbench/composer_prompt.txt'
with open(file_path, 'r') as file:
    # Read the entire content of the file
    user1 = file.read()
new_query = '# Query:' + user1.split('# Query:')[-1]
user1 = ''.join(user1.split('# Query:')[:-1]).strip()
system_query = "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."
user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."

# chat1 = model.start_chat(history=[])
# response = chat1.send_message(system_query, 
#                               generation_config = text_config,
#                               safety_settings=safety_settings)
# response1 = chat1.send_message(user1,
#                                 generation_config = text_config,
#                                 safety_settings=safety_settings)
# response2 = chat1.send_message("# Query: pour the contents of the red mug into the maroon one",
#                                 generation_config = text_config,
#                                 safety_settings=safety_settings)
# ret = response2.text
# print(ret)

# # for message in chat1.history:
# #     print(f'**{message.role}**: {message.parts[0].text}')
# print("*"*50)

# chat1 = model.start_chat(history=[])
response2 = model.generate_content(f"{system_query}\n{user1}\n# Query: pour the contents of the red mug into the maroon one",
                                    generation_config = text_config,
                                    safety_settings=safety_settings)
print(response2.text)
# for message in chat1.history:
#     print(f'**{message.role}**: {message.parts[0].text}')
# print("*"*50)
# chat2 = model.start_chat(history=[])
# response = chat2.send_message("Which is india's capital?")
# response1 = chat2.send_message("What about it's population?")
# for message in chat2.history:
#     print(f'**{message.role}**: {message.parts[0].text}')
# print("*"*50)
# chat3 = model.start_chat(history=chat1.history)
# response = chat3.send_message("what's the name of it's currency?")
# for message in chat3.history:
#     print(f'**{message.role}**: {message.parts[0].text}')