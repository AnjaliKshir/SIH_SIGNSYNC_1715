from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import json
import ssl
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import ParentedTree
import stanza
import pprint
from nltk.tree import Tree
from googletrans import Translator
from langdetect import detect

from pymongo import MongoClient
import gridfs

# new project

ssl._create_default_https_context = ssl._create_unverified_context

# Ensure you have the NLTK data downloaded
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

translator = Translator()

# Pipeline for stanza (calls spacy for tokenizer)
stanza.download('en', model_dir='stanza_resources')
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})

# Stop words that are not to be included in ISL
stop_words = set(["am", "are", "is", "was", "were", "be", "being", "been", "have", "has", "had",
                  "does", "did", "could", "should", "would", "can", "shall", "will", "may", "might", "must", "let", "do", 'to'])

# Global variables for processing
sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []
final_words = []
final_words_detailed = []
final_output_in_sent = []
final_words_dict = {}

def clear_all():
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words.clear()
    final_words_detailed.clear()
    final_output_in_sent.clear()
    global final_words_dict
    final_words_dict = {}  # Re-initialize the dict

def convert_to_sentence_list(text):
    for sentence in text.sentences:
        sent_list.append(sentence.text)
        sent_list_detailed.append(sentence)

def convert_to_word_list(sentences):
    temp_list = []
    temp_list_detailed = []
    for sentence in sentences:
        for word in sentence.words:
            temp_list.append(word.text)
            temp_list_detailed.append(word)
        word_list.append(temp_list.copy())
        word_list_detailed.append(temp_list_detailed.copy())
        temp_list.clear()
        temp_list_detailed.clear()

def filter_words(word_list):
    temp_list = []
    final_words = []
    for words in word_list:
        temp_list.clear()
        for word in words:
            if word not in stop_words:
                temp_list.append(word)
        final_words.append(temp_list.copy())
    for words in word_list_detailed:
        for i, word in enumerate(words):
            if words[i].text in stop_words:
                del words[i]
                break
    return final_words

def remove_punct(word_list):
    for words, words_detailed in zip(word_list, word_list_detailed):
        for i, (word, word_detailed) in enumerate(zip(words, words_detailed)):
            if word_detailed.upos == 'PUNCT':
                del words_detailed[i]
                words.remove(word_detailed.text)
                break

def lemmatize(final_word_list):
    for words, final in zip(word_list_detailed, final_word_list):
        for i, (word, fin) in enumerate(zip(words, final)):
            if fin in word.text:
                if len(fin) == 1:
                    final[i] = fin
                else:
                    final[i] = word.lemma
    for word in final_word_list:
        print("final_words", word)

def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}
    for sub_tree in parent_tree.subtrees():
        if sub_tree is not None:
            tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag

def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    if sub_tree is not None and sub_tree.parent() is not None:
        if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
            tree_traversal_flag[sub_tree.treeposition()] = 1
            modified_parse_tree.insert(i, sub_tree)
            i += 1
    return i, modified_parse_tree

def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i += 1
    return i, modified_parse_tree

def modify_tree_structure(parent_tree):
    if parent_tree is None:
        print("Parent tree is None.")
        return Tree('ROOT', [])

    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree is None:
            print("Sub-tree is None.")
            continue
        print("Sub-tree:", sub_tree)
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        elif sub_tree.label() in ["VP", "PRP"]:
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if child_sub_tree is None:
                print("Child sub-tree is None.")
                continue
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag.get(child_sub_tree.treeposition(), 0) == 0 and \
                   tree_traversal_flag.get(child_sub_tree.parent().treeposition(), 0) == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i += 1

    return modified_parse_tree



def reorder_eng_to_isl(input_string):
    parser = CoreNLPParser(url='http://localhost:9000')  # Ensure CoreNLP server is running
    try:
        parse_tree = next(parser.raw_parse(input_string))
        if parse_tree is None:
            print("Parse tree is None.")
            return input_string.split()  # Return words as they are if parsing fails
        parent_tree = ParentedTree.convert(parse_tree)
        modified_parse_tree = modify_tree_structure(parent_tree)
        parsed_sent = modified_parse_tree.leaves()
        return parsed_sent
    except Exception as e:
        print(f"Error parsing input string: {str(e)}")
        return input_string.split()


def pre_process(text):
    remove_punct(word_list)
    final_words.extend(filter_words(word_list))
    lemmatize(final_words)

def final_output(input):
    final_string = ""
    valid_words = open("words.txt", 'r').read()
    valid_words = valid_words.split('\n')
    fin_words = []
    for word in input:
        word = word.lower()
        if word not in valid_words:
            for letter in word:
                fin_words.append(letter)
        else:
            fin_words.append(word)
    return fin_words

def convert_to_final():
    for words in final_words:
        final_output_in_sent.append(final_output(words))

def take_input(text):

    test_input = text.strip().replace("\n", "").replace("\t", "")
    test_input2 = ""
    if len(test_input) == 1:
        test_input2 = test_input
    else:
        for word in test_input.split("."):
            test_input2 += word.capitalize() + " ."
    some_text = en_nlp(test_input2)
    convert(some_text)

def convert(some_text):
    convert_to_sentence_list(some_text)
    convert_to_word_list(sent_list_detailed)
    for i, words in enumerate(word_list):
        word_list[i] = reorder_eng_to_isl(" ".join(words))
    pre_process(some_text)
    convert_to_final()
    remove_punct(final_output_in_sent)
    print_lists()

def print_lists():
    print("--------------------Word List------------------------")
    pprint.pprint(word_list)
    print("--------------------Final Words------------------------")
    pprint.pprint(final_words)
    print("---------------Final sentence with letters--------------")
    pprint.pprint(final_output_in_sent)





@app.route('/', methods=['GET'])
def index():
    clear_all()
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
async def process_text():
    try:
        clear_all()

        data = request.get_json()
        print(f"Received data: {data}")
        user_text = data.get('text',"")

        if not user_text:
            print("No text provided.")
            return jsonify({'error': 'No text provided'}), 400
        
        print("Text before processing:", user_text)

        try:
            # Detect the language of the input text
            detected_lang = detect(user_text)
            print(f"Detected language: {detected_lang}")

            if detected_lang != 'en':
                translated_text = await translator.translate(user_text, src=detected_lang, dest='en')
                # print(f"Translated Text: {translated_text.pronunciation}")  # Debugging
                response = jsonify({"translated_text": translated_text.text})
                print(response.get_json())
                take_input(translated_text.text)

            else:
                translated_text = user_text
                print("Input text is already in English.")
                take_input(translated_text)

        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        for words in final_output_in_sent:
            for i, word in enumerate(words, start=1):
                final_words_dict[i] = word
        print("Final words dict:", final_words_dict)
        animations = {}
        for key, word in final_words_dict.items():
            word = word.lower()
            animation_path = os.path.join('static/', f'{word}.mp4')
            if os.path.exists(animation_path):
                animations[key] = word
        return jsonify({"modified_text": animations})
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

