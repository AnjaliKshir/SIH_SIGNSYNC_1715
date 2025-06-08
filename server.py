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
# from gridfs import GridFS
from io import BytesIO
from dotenv import load_dotenv

import time

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading

import io
from google.cloud import speech
from google.cloud import translate_v2 as translate

# Set your GCP credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "cogent-case-460012-s9-9ea78606f876.json")

# Init Google APIs
speech_client = speech.SpeechClient()
translate_client = translate.Client()

# Load environment variables from the .env file
load_dotenv()

# new project

ssl._create_default_https_context = ssl._create_unverified_context

# Ensure you have the NLTK data downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

translator = Translator()

# Get MongoDB URL from environment variables
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise Exception("MONGO_URL not found in .env file")

# MongoDB connection
client = MongoClient(mongo_url)
db = client["ISL_vdoDB"]
collection = db["videos"]
fs = gridfs.GridFS(db)


# Pipeline for stanza (calls spacy for tokenizer)
stanza.download('en', model_dir='stanza_resources')
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})

# Stop words that are not to be included in ISL
stop_words = set(["a", "am", "are", "is", "was", "were", "be", "being", "been", "have", "has", "had",
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

    # First, add all words in their original order
    original_words = parent_tree.leaves()
    i = 0
    for word in original_words:
        leaf_tree = None
        # Find the subtree that contains just this word
        for sub_tree in parent_tree.subtrees():
            if len(sub_tree.leaves()) == 1 and sub_tree.leaves()[0] == word:
                leaf_tree = sub_tree
                break
        
        if leaf_tree:
            tree_traversal_flag[leaf_tree.treeposition()] = 1
            modified_parse_tree.insert(i, leaf_tree)
        else:
            # If we can't find the subtree, just add the word directly
            word_tree = Tree('WORD', [word])
            modified_parse_tree.insert(i, word_tree)
        i += 1
    
    return modified_parse_tree

    # i = 0
    # for sub_tree in parent_tree.subtrees():
    #     if sub_tree is None:
    #         print("Sub-tree is None.")
    #         continue
    #     print("Sub-tree:", sub_tree)
    #     if sub_tree.label() == "NP":
    #         i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
    #     elif sub_tree.label() in ["VP", "PRP"]:
    #         i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    # for sub_tree in parent_tree.subtrees():
    #     for child_sub_tree in sub_tree.subtrees():
    #         if child_sub_tree is None:
    #             print("Child sub-tree is None.")
    #             continue
    #         if len(child_sub_tree.leaves()) == 1:
    #             if tree_traversal_flag.get(child_sub_tree.treeposition(), 0) == 0 and \
    #                tree_traversal_flag.get(child_sub_tree.parent().treeposition(), 0) == 0:
    #                 tree_traversal_flag[child_sub_tree.treeposition()] = 1
    #                 modified_parse_tree.insert(i, child_sub_tree)
    #                 i += 1

    # return modified_parse_tree



def reorder_eng_to_isl(input_string):
    """
    Enhanced ISL grammar reordering function that handles:
    - Basic SOV structure
    - Questions (WH-questions and Yes/No questions)
    - Complex sentences with conjunctions
    - Time expressions
    - Negations
    - Modifiers and adjectives
    """
    # Special handling for greetings and simple phrases
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                "goodbye", "bye", "welcome", "greetings", "namaste"]
    
    words = input_string.split()
    if not words:
        return words

    # Handle greetings
    greeting_words = []
    remaining_words = words.copy()
    
    # Identify greeting word(s) at the beginning
    first_word_lower = words[0].lower()
    if first_word_lower in greetings:
        greeting_words = [words[0]]
        remaining_words = words[1:]
    elif len(words) >= 2 and " ".join(words[:2]).lower() in greetings:
        greeting_words = words[:2]
        remaining_words = words[2:]

    if not remaining_words:
        return greeting_words

    try:
        # Parse the sentence
        parser = CoreNLPParser(url='http://localhost:9000')
        parse_tree = next(parser.raw_parse(" ".join(remaining_words)))
        if parse_tree is None:
            return greeting_words + apply_enhanced_sov(remaining_words)
        
        parent_tree = ParentedTree.convert(parse_tree)
        
        # Handle different sentence types
        if is_question(remaining_words):
            return greeting_words + handle_question(remaining_words, parent_tree)
        elif is_negative(remaining_words):
            return greeting_words + handle_negative(remaining_words, parent_tree)
        elif is_complex_sentence(remaining_words):
            return greeting_words + handle_complex_sentence(remaining_words, parent_tree)
        else:
            modified_parse_tree = modify_tree_structure(parent_tree)
            parsed_sent = modified_parse_tree.leaves()
            return greeting_words + parsed_sent

    except Exception as e:
        print(f"Error parsing input string: {str(e)}")
        return greeting_words + apply_enhanced_sov(remaining_words)

def is_question(words):
    """Check if the sentence is a question."""
    question_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
    return (words[0].lower() in question_words or 
            words[-1].endswith('?') or 
            any(word.lower() in ["is", "are", "do", "does", "did", "can", "could", "will", "would"] for word in words[:2]))

def is_negative(words):
    """Check if the sentence contains negation."""
    negations = ["not", "no", "never", "neither", "nor", "none", "nothing", "nowhere"]
    return any(word.lower() in negations for word in words)

def is_complex_sentence(words):
    """Check if the sentence is complex (contains conjunctions, relative clauses, or conditionals)."""
    conjunctions = ["and", "or", "but", "because", "although", "while", "since", "unless", "if"]
    relative_pronouns = ["who", "whom", "whose", "which", "that"]
    conditional_words = ["if", "unless", "provided", "assuming", "supposing"]
    
    return any(word.lower() in conjunctions + relative_pronouns + conditional_words for word in words)

def handle_question(words, parse_tree):
    """Handle different types of questions in ISL."""
    # WH-questions
    wh_words = ["what", "who", "where", "when", "why", "how", "which", "whose"]
    if words[0].lower() in wh_words:
        # Move WH-word to the end
        return words[1:] + [words[0]]
    
    # Yes/No questions
    if words[0].lower() in ["is", "are", "do", "does", "did", "can", "could", "will", "would"]:
        # Move auxiliary verb to the end
        return words[1:] + [words[0]]
    
    # Questions ending with '?'
    if words[-1].endswith('?'):
        words[-1] = words[-1].rstrip('?')
        return words

def handle_negative(words, parse_tree):
    """Handle negative sentences in ISL."""
    negations = ["not", "no", "never", "neither", "nor", "none", "nothing", "nowhere"]
    neg_word = next(word for word in words if word.lower() in negations)
    neg_index = words.index(neg_word)
    
    # Move negation to the end
    return words[:neg_index] + words[neg_index+1:] + [neg_word]

def handle_complex_sentence(words, parse_tree):
    """Enhanced handling of complex sentences with various structures."""
    # Check for relative clauses
    if has_relative_clause(words):
        return handle_relative_clause(words, parse_tree)
    
    # Check for conditional sentences
    if has_conditional(words):
        return handle_conditional(words, parse_tree)
    
    # Check for passive voice
    if is_passive_voice(words):
        return handle_passive_voice(words, parse_tree)
    
    # Handle regular complex sentences with conjunctions
    conjunctions = ["and", "or", "but", "because", "although", "while", "since", "unless", "if"]
    conj_word = next((word for word in words if word.lower() in conjunctions), None)
    
    if conj_word:
        conj_index = words.index(conj_word)
        first_part = words[:conj_index]
        second_part = words[conj_index+1:]
        
        # Process each part separately
        first_processed = apply_enhanced_sov(first_part)
        second_processed = apply_enhanced_sov(second_part)
        
        # Combine with conjunction at the end
        return first_processed + second_processed + [conj_word]
    
    return apply_enhanced_sov(words)

def has_relative_clause(words):
    """Check if the sentence contains a relative clause."""
    relative_pronouns = ["who", "whom", "whose", "which", "that"]
    return any(word.lower() in relative_pronouns for word in words)

def handle_relative_clause(words, parse_tree):
    """Handle sentences with relative clauses."""
    relative_pronouns = ["who", "whom", "whose", "which", "that"]
    
    # Find the relative pronoun
    rel_pronoun = next(word for word in words if word.lower() in relative_pronouns)
    rel_index = words.index(rel_pronoun)
    
    # Split into main clause and relative clause
    main_clause = words[:rel_index]
    relative_clause = words[rel_index+1:]
    
    # Process each part
    main_processed = apply_enhanced_sov(main_clause)
    relative_processed = apply_enhanced_sov(relative_clause)
    
    # In ISL, relative clauses often come after the main clause
    return main_processed + [rel_pronoun] + relative_processed

def has_conditional(words):
    """Check if the sentence is conditional."""
    conditional_words = ["if", "unless", "provided", "assuming", "supposing"]
    return any(word.lower() in conditional_words for word in words)

def handle_conditional(words, parse_tree):
    """Handle conditional sentences."""
    conditional_words = ["if", "unless", "provided", "assuming", "supposing"]
    
    # Find the conditional word
    cond_word = next(word for word in words if word.lower() in conditional_words)
    cond_index = words.index(cond_word)
    
    # Split into condition and result
    condition = words[cond_index+1:]
    result = words[:cond_index]
    
    # Process each part
    condition_processed = apply_enhanced_sov(condition)
    result_processed = apply_enhanced_sov(result)
    
    # In ISL, condition often comes after the result
    return result_processed + [cond_word] + condition_processed

def is_passive_voice(words):
    """Check if the sentence is in passive voice."""
    passive_indicators = ["is", "are", "was", "were", "be", "been", "being"]
    return any(word.lower() in passive_indicators for word in words)

def handle_passive_voice(words, parse_tree):
    """Handle passive voice sentences."""
    passive_indicators = ["is", "are", "was", "were", "be", "been", "being"]
    
    # Find the passive indicator
    passive_word = next(word for word in words if word.lower() in passive_indicators)
    passive_index = words.index(passive_word)
    
    # Split into subject and predicate
    subject = words[:passive_index]
    predicate = words[passive_index+1:]
    
    # Process each part
    subject_processed = apply_enhanced_sov(subject)
    predicate_processed = apply_enhanced_sov(predicate)
    
    # In ISL, passive voice often follows a different structure
    return subject_processed + predicate_processed + [passive_word]

def apply_enhanced_sov(words):
    """Enhanced SOV ordering with better handling of modifiers, time expressions, and adverbial phrases."""
    if not words:
        return words
    
    # Remove auxiliary verbs
    aux_verbs = ["is", "are", "am", "was", "were", "do", "does", "did", 
                "has", "have", "had", "will", "shall", "should", "would", 
                "can", "could", "may", "might", "must"]
    
    filtered_words = [w for w in words if w.lower() not in aux_verbs]
    
    # Handle time expressions
    time_words = ["today", "tomorrow", "yesterday", "now", "then", "before", "after"]
    time_expressions = []
    
    # Handle adverbial phrases
    adverbs = ["quickly", "slowly", "carefully", "well", "badly", "easily", "hard"]
    adverbial_phrases = []
    
    # Handle prepositional phrases
    prepositions = ["in", "on", "at", "by", "with", "to", "from", "of", "for"]
    prepositional_phrases = []
    
    # Handle comparative and superlative structures
    comparatives = ["more", "less", "better", "worse", "bigger", "smaller"]
    superlatives = ["most", "least", "best", "worst", "biggest", "smallest"]
    
    remaining_words = []
    
    for word in filtered_words:
        if word.lower() in time_words:
            time_expressions.append(word)
        elif word.lower() in adverbs:
            adverbial_phrases.append(word)
        elif word.lower() in prepositions:
            prepositional_phrases.append(word)
        elif word.lower() in comparatives + superlatives:
            remaining_words.append(word)
        else:
            remaining_words.append(word)
    
    # Basic SOV ordering for remaining words
    if len(remaining_words) >= 3:
        subject = remaining_words[:1]
        verb = remaining_words[-1:]
        object_words = remaining_words[1:-1]
        
        # Combine all elements in ISL order
        return (time_expressions + 
                subject + 
                object_words + 
                verb + 
                adverbial_phrases + 
                prepositional_phrases)
    
    return (time_expressions + 
            remaining_words + 
            adverbial_phrases + 
            prepositional_phrases)

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
    # Clean the input text
    test_input = text.strip().replace("\n", "").replace("\t", "")

    # # Better sentence handling
    # test_input2 = ""
    # if len(test_input) == 1:
    #     test_input2 = test_input
    # else:
    #     # Split by sentence-ending punctuation but keep the punctuation
    #     import re
    #     sentences = re.split('([.!?])', test_input)
    #     test_input2 = ""
    #     for i in range(0, len(sentences)-1, 2):
    #         if sentences[i].strip():  # Skip empty strings
    #             test_input2 += sentences[i].strip().capitalize() + sentences[i+1] + " "
    # # else:
    # #     for word in test_input.split("."):
    # #         test_input2 += word.capitalize() + " ."
    
    # some_text = en_nlp(test_input2)

    some_text = en_nlp(test_input)
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

def cleanup_temp_videos():
    temp_dir = "temp_videos"
    current_time = time.time()

    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > 86400:  # 24 hours in seconds
                os.remove(file_path)
                print(f"Deleted old temp video: {file_path}")




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
            # Detect language using Google Translate
            detection = translate_client.detect_language(user_text)
            detected_lang = detection['language']
            print(f"Detected language: {detected_lang}")

            if detected_lang != 'en':
                translation = translate_client.translate(user_text, target_language='en')
                translated_text = translation['translatedText']
                print(f"Translated Text: {translated_text}")
                take_input(translated_text)
            else:
                translated_text = user_text
                print("Input text is already in English.")
                take_input(translated_text)


        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        # for words in final_output_in_sent:
        #     for i, word in enumerate(words, start=1):
        #         final_words_dict[i] = sword

        index = 1  # Global counter
        for words in final_output_in_sent:
            for word in words:
                final_words_dict[index] = word
                index += 1  # Increment for each word across all sentences


        print("Final words dict:", final_words_dict)
        animations = {}
        for key, word in final_words_dict.items():
            word = word.lower()
            
            # # Fetch the video from MongoDB
            # file = fs.find_one({"word": word})
            # if file:
            #     # Save the video temporarily to serve it
            #     temp_video_path = os.path.join('static','temp_videos', f'{word}.mp4')
                
            #     # Ensure the temporary directory exists
            #     os.makedirs('static','temp_videos', exist_ok=True)
                
            #     with open(temp_video_path, 'wb') as temp_video_file:
            #         temp_video_file.write(file.read())

            #     # animations[key] = temp_video_path  # Store the path to serve or use later
            #     animations[key] = word
            # else:
            #     print(f"No video found for the word: {word}")

            animation_path = os.path.join('static/', f'{word}.mp4')
            if os.path.exists(animation_path):
                animations[key] = word
        return jsonify({"modified_text": animations})
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


# logic for gesture recognition

is_running = False
thread = None

# Global variables for OpenCV
cap = None
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["Bye", "Good", "Hello", "No", "Okay", "ThankYou", "Yes"]

def run_hand_detection():
    global cap, is_running
    try:
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize frame counter and timestamp
        frame_count = 0
        last_timestamp = 0

        while is_running:
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame from camera")
                break

            try:
                # Increment frame counter
                frame_count += 1
                current_timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                
                # Skip frames if processing is too slow
                if current_timestamp - last_timestamp < 33:  # ~30 FPS
                    continue
                
                last_timestamp = current_timestamp
                
                imgOutput = img.copy()
                hands, img = detector.findHands(img, draw=False)  # Disable drawing to improve performance
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    
                    # Ensure crop coordinates are within bounds
                    x1 = max(0, x - offset)
                    y1 = max(0, y - offset)
                    x2 = min(img.shape[1], x + w + offset)
                    y2 = min(img.shape[0], y + h + offset)
                    
                    imgCrop = img[y1:y2, x1:x2]
                    
                    if imgCrop.size > 0:
                        aspectRatio = h / w
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize
                        
                        try:
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)
                            
                            # Draw results on output image
                            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(imgOutput, labels[index], (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error in classification: {str(e)}")
                            continue
                
                try:
                    cv2.imshow('Image', imgOutput)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    print(f"Display error: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in hand detection: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

@app.route('/start', methods=['POST'])
def start_script():
    global is_running, thread
    if not is_running:
        try:
            is_running = True
            thread = threading.Thread(target=run_hand_detection)
            thread.daemon = True  # Make thread daemon so it exits when main program exits
            thread.start()
            return jsonify({"status": "Started"})
        except Exception as e:
            is_running = False
            return jsonify({"status": "Error", "message": str(e)}), 500
    return jsonify({"status": "Already running"})

@app.route('/stop', methods=['POST'])
def stop_script():
    global is_running, cap
    if is_running:
        is_running = False
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        return jsonify({"status": "Stopped"})
    return jsonify({"status": "Not running"})

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_content = audio_file.read()

    # Convert audio for Google Speech-to-Text
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,  # or 16000 based on how you capture
        language_code="hi-IN",  # can be auto-detected, but start with Hindi as example
        alternative_language_codes=["en-US"],  # Also try English fallback
        # enable_automatic_punctuation=True
    )

    response = speech_client.recognize(config=config, audio=audio)

    if not response.results:
        return jsonify({"transcript": "", "error": "No speech recognized"}), 200

    transcript = response.results[0].alternatives[0].transcript
    print("Transcript:", transcript)

    # Detect & translate if not in English
    translation = translate_client.translate(transcript, target_language="en")
    translated_text = translation["translatedText"]
    print("Translated:", translated_text)

    return jsonify({"transcript": translated_text})

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        width = data['width']
        height = data['height']
        frame_data = np.array(data['data'], dtype=np.uint8).reshape((height, width, 4))
        
        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2BGR)
        
        # Process frame with hand detector
        hands, frame = detector.findHands(frame, draw=False)  # Disable drawing to improve performance
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Ensure crop coordinates are within bounds
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(frame.shape[1], x + w + offset)
            y2 = min(frame.shape[0], y + h + offset)
            
            imgCrop = frame[y1:y2, x1:x2]
            
            if imgCrop.size > 0:
                aspectRatio = h / w
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                try:
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    return jsonify({
                        'detected': True,
                        'label': labels[index],
                        'bbox': {
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    })
                except Exception as e:
                    print(f"Error in classification: {str(e)}")
                    return jsonify({'detected': False, 'error': str(e)})
        
        return jsonify({'detected': False})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({
            'detected': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
