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
from stopping import MyStoppingCriteria
from llama_cpp import LlamaTokenizer

app = Flask(__name__)

tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf", token='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')

N_CTX=1400

LLM = Llama(model_path='LLM/ggml-model-Q4_K.gguf', n_gpu_layers=-1, n_ctx=2048)

def summarize(input_text):
    prompt = f"""Genera un riassunto in italiano del testo seguente. Voglio soltanto il riassunto senza tue introduzioni o considerazioni.\
         è molto importante che il tuo output sia soltanto il riassunto come se fosse un articolo, non simulare una risposta. Voglio soltanto una risposta dell'Assistente.\
         Il riassunto deve essere in lingua italiana. Voglio soltanto un input dell Umano (il mio) e una risposta dell'Assistente.
[|Umano|] {input_text}"""
# [|Assistente|]"""
    # output = LLM(prompt, max_tokens=0, stopping_criteria=MyStoppingCriteria("[|Umano|]", prompt))
    # res = output['choices'][0]['text']
    # return res
    output = tokenizer.decode(output.sequences[0]).split("[|AI|]")[1]
    return output
    

def wiki_summ(url):
    try:
        # if datetime.now() > datetime(2024,2,7,17,50):
        #     print('din don')
        #     os._exit(0)
        json_data = ''
        response = requests.get(url=url,)
        soup = BeautifulSoup(response.content, 'html.parser')
        heading = soup.find(id="firstHeading")
        title = heading.text
        print(f'Working {title}')
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
            sum_part = summarize(t)
            summ += sum_part
        if text == '':
            print('no text')
            return
        summ = clean_summ(summ)
        doub_summ = summarize(summ)
        if '\n' in summ[:50]:
            summ = summ.split('\n')[1]
        if '\n' in doub_summ[:50]:
            doub_summ = doub_summ.split('\n', 1)[1]
        elapsed = (datetime.now() - t1)
        json_data = {"title": title, "text": text.strip(), "summary": summ.strip(), "double_summary": doub_summ.strip(), "time": elapsed}
        json_title = title.replace(' ', '_').replace("'", "")
        with open (f'riassunti/{json_title}.json', 'w') as f:
            json.dump(json_data, f, default=str)
        print(f'Finished {title}')
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
        wiki_summ("https://it.wikipedia.org" + linkToScrape['href'])
    except Exception as e:
        print(e)

def clean_summ(summ):
#     prompt = f"[|Umano|]Il tuo compito è quello di generare un riassunto del testo seguente: {summ}.\
# Genera solo il riassunto senza spiegazioni. Interrompi la generazione del testo quando scrivi |Umano|. [|Assistente|]"
    prompt = f"""Pulisci, correggi e migliora questo testo. Devi presentarlo come se fossi una guida turistica. Voglio soltanto una risposta dell'Assistente.
[|Umano|] {summ}"""
#  [|Assistente|]"""
    output = LLM(prompt, max_tokens=0, stopping_criteria=MyStoppingCriteria("[|Umano|]", prompt))
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
    wiki_summ("https://it.wikipedia.org/wiki/Neptune_(vascello)")
    # double_summ()
    # chat()
    # print('App served on port 9999')
    # serve(app, port=9999)