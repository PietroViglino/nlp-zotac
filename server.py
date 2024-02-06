from flask import Flask, jsonify, request
from waitress import serve
from llama_cpp import Llama
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import random

app = Flask(__name__)

# LLM = Llama(model_path="LLM/llama-2-13b-chat.Q5_K_M.gguf", n_ctx=4096)

# LLM = Llama(model_path="LLM/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=4096)
# LLM  = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=4096, verbose=True) # 4:.. con test.txt
LLM = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=2048) # 0:19 con test.txt   <---- best token/speed
# LLM = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=1280, n_batch=64) # 0:12  con test.txt

@app.route('/api/summarize')
def summarize(input_text=None):
    print('Starting elaboration of summary')
    if input_text is None:
        input_text = request.json.get("text")
    system_message = "Generate a summary of the following text. What i want to obtain is just the summary without response's introduction or considerations. I want just the summary.\
        Avoid at all cost sentences outside the summary, like this one: 'Here is the summary of the text:\n'"
    user_message = input_text
    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {user_message} [/INST]"""
    output = LLM(prompt, max_tokens=0)
    print('Done')
    res = output['choices'][0]['text']
    if '\n' in res:
        res = res.split('\n')[-1]
    if input_text == None:
        return jsonify(output["choices"][0]["text"])
    elif input_text != None:
        return res

def wiki_summ(url):
    try:
        json_data = ''
        response = requests.get(url=url,)
        soup = BeautifulSoup(response.content, 'html.parser')
        heading = soup.find(id="firstHeading")
        title = heading.text
        text = []
        if title != 'Web scraping' and  ':' not in title:  
            parafs = soup.find_all("p", recursive=True)
            for p in parafs:
                p_content = p.text
                p_content = p_content.replace('\n', '')
                p_content = re.sub(r'\[[^\]]*\]', '', p_content)
                # p_content = re.sub(r'\([^)]*\)', '', p_content)
                p_list = re.split(r'(?<![A-Z])\.\s', p_content)
                for sentence in p_list:
                    if sentence != '':
                        if sentence[-1] != '.':
                            sentence += '.'
                        text.append(sentence)
        text = ' \n'.join(text)
        summ = summarize(text)
        summ = ' \n'.join(summ.split('.'))
        json_data = f'title: {title},\ntext: {text},\nsummary: {summ}\n\n'
        with open('wiki_summ.txt', 'a') as f:
            f.write(json_data)
        # get next link to scrape
        allLinks = soup.find(id="bodyContent").find_all("a")
        random.shuffle(allLinks)
        linkToScrape = 0
        for link in allLinks:
            if link['href'].find("/wiki/") == -1:
                continue
            linkToScrape = link
            break
        time.sleep(1)
        wiki_summ("https://en.wikipedia.org" + linkToScrape['href'])
    except Exception as e:
        print(e)
        wiki_summ("https://en.wikipedia.org/wiki/Web_scraping")

wiki_summ("https://en.wikipedia.org/wiki/Web_scraping")

# while True:
#     user_input = input('\nInput: ')
#     system_message = "You are an assistant."
#     prompt = f"""<s>[INST] <<SYS>>
#     {system_message}
#     <</SYS>>
#     {user_input} [/INST]"""
#     output = LLM(prompt, max_tokens=0)
#     print('\n' + output['choices'][0]['text'])

if __name__ == '__main__':
    print('App served on port 9999')
    serve(app, port=9999)