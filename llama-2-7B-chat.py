from flask import Flask, jsonify, request
from waitress import serve
from llama_cpp import Llama
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import os
from datetime import datetime
from translate import Translator
from deep_translator import GoogleTranslator

app = Flask(__name__)
translator= Translator(to_lang="it")

N_CTX=1200

# LLM = Llama(model_path="LLM/llama-2-13b-chat.Q5_K_M.gguf", n_ctx=4096)

# LLM = Llama(model_path="LLM/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=4096)
# LLM  = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=4096, verbose=True) # 4:.. con test.txt
LLM = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=2048) # 0:19 con test.txt   <---- best token/speed
# LLM = Llama(model_path='LLM/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=1280, n_batch=64) # 0:12  con test.txt

def split_text(text, n):
  chunks = text.split()
  punctuation = [".", "!", "?", "..."]
  for i in range(len(chunks) - 1, -1, -1):
    if chunks[i] in punctuation:
      chunks.insert(i + 1, chunks[i])
      chunks.pop(i)
  current_chunk = ""
  for chunk in chunks:
    if len(current_chunk) + len(chunk) + 1 <= n:
      current_chunk += " " + chunk
    else:
      if current_chunk:
        yield current_chunk.strip()
      current_chunk = chunk
  if current_chunk:
    yield current_chunk.strip()

@app.route('/api/simple_summ')
def simple_summ(input_text=None):
    print('Starting elaboration of summary')
    if input_text is None:
        input_text = request.json.get("text")
    system_message = "Generate a summary of the following text.\
        Do never make references to Wikipedia.\
        You should present the cleaned content as if you were a touristic guide.\
        Avoid any kind of introduction."
    user_message = input_text
    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {user_message} [/INST]"""
    output = LLM(prompt, max_tokens=0)
    res = output['choices'][0]['text']
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
        print(f'Processing {title}')
        text = []
        parafs = soup.find_all("p", recursive=True)
        for p in parafs:
            p_content = p.text
            p_content = p_content.replace('\n', '')
            p_content = re.sub(r'\[[^\]]*\]', '', p_content)
            p_content = re.sub(r'\([^)]*\)', '', p_content)
            p_list = re.split(r'(?<![A-Z])\.\s', p_content)
            for sentence in p_list:
                if sentence != '':
                    if sentence[-1] != '.':
                        sentence += '.'
                    text.append(sentence)
        text = ' '.join(text)
        t1 = datetime.now()
        text_b = bytes(text, 'utf-8')
        tokens = LLM.tokenize(text_b)
        splitted_tokens = [tokens[i:i+N_CTX] for i in range(0, len(tokens), N_CTX)]
        summ = ''
        for t in splitted_tokens:
            t = LLM.detokenize(t)
            sum_part = simple_summ(t)
            summ += sum_part
        if text == '':
            print('no text')
            return
        summ = clean_summ(summ)
        elapsed1 = (datetime.now() - t1)
        doub_summ = simple_summ(summ)
        if '\n' in summ[:50]:
            summ = summ.split('\n')[1]
        if '\n' in doub_summ[:50]:
            doub_summ = doub_summ.split('\n', 1)[1]
        elapsed2 = (datetime.now() - t1)
        json_data = {"title": title, "text": text.strip(), "summary": summ.strip(), "double_summary": doub_summ.strip(), "time1": elapsed1, "time2": elapsed2}
        json_title = title.replace(' ', '_').replace("'", "")
        with open (f'summaries/{json_title}.json', 'w') as f:
            json.dump(json_data, f, default=str)
        print(f'Finished {title}')
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

@app.route('/api/summarize_trad_1')
def summarize_t1():
    t_0 = datetime.now()
    text = request.json.get("text")
    text_b = bytes(text, 'utf-8')
    tokens = LLM.tokenize(text_b)
    splitted_tokens = [tokens[i:i+N_CTX] for i in range(0, len(tokens), N_CTX)]
    summ = ''
    summ_t = []
    for t in splitted_tokens:
        t = LLM.detokenize(t)
        sum_part = simple_summ(t) + ' '
        summ += sum_part
        summ_t.append(str(datetime.now() - t_0))
    if text == '':
        print('no text')
        return
    summ = clean_summ(summ)
    summ = summ.split('\n')
    summ = ''.join(summ[1:])
    trad = ''
    splitted_summ = split_text(summ, 500)
    for block in splitted_summ:
        trad += translator.translate(block) + ' '
    trad_t = str(datetime.now() - t_0)
    res = {"summ": summ, "summ_t": summ_t, "trad": trad, "trad_t": trad_t}
    return jsonify(res)

@app.route('/api/summarize_trad_2')
def summarize_t2():
    t_0 = datetime.now()
    text = request.json.get("text")
    text_b = bytes(text, 'utf-8')
    tokens = LLM.tokenize(text_b)
    splitted_tokens = [tokens[i:i+N_CTX] for i in range(0, len(tokens), N_CTX)]
    summ = ''
    summ_t = []
    for t in splitted_tokens:
        t = LLM.detokenize(t)
        sum_part = simple_summ(t) + ' '
        summ += sum_part
        summ_t.append(str(datetime.now() - t_0))
    if text == '':
        print('no text')
        return
    summ = clean_summ(summ)
    summ = summ.split('\n')
    summ = ''.join(summ[1:])
    trad = ''
    splitted_summ = split_text(summ, 500)
    for block in splitted_summ:
        trad += GoogleTranslator(source='en', target='it').translate(block) + ' '
    trad_t = str(datetime.now() - t_0)
    res = {"summ": summ, "summ_t": summ_t, "trad": trad, "trad_t": trad_t}
    return jsonify(res)

def clean_summ(summ):
    system_message = "Please clean the following text, as it is now badly written with some mistakes and repetitions.\
        It is very important that you only output the cleaned text as if it was an article, you shouldn't simulate an answer.\
        Omit everything that isn't the cleaned text, like your introduction, conclusion or your notes.\
        Do never make references to Wikipedia.\
        You should present the cleaned content as if you were a touristic guide.\
        Avoid any kind of introduction and be serious."
    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {summ} [/INST]"""
    output = LLM(prompt, max_tokens=0)
    res = output['choices'][0]['text']
    return res
           
def chat():
    while True:
        user_input = input('\nInput: ')
        system_message = "You are an assistant."
        prompt = f"""<s>[INST] <<SYS>>
        {system_message}
        <</SYS>>
        {user_input} [/INST]"""
        output = LLM(prompt, max_tokens=0)
        print('\n' + output['choices'][0]['text'])

if __name__ == '__main__':
    # wiki_summ("https://en.wikipedia.org/wiki/Design")
    # double_summ()
    # chat()
    print('App served on port 9999')
    serve(app, port=9999)