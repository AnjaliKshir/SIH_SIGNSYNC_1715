Steps to execute:
1. Install all the packages mentioned in requirements.txt 
    - pip install -r requirements.txt
2. Run python server in root directory 
    - python server.py
3. Run Java NLP server in "cd .\stanford-corenlp-4.5.7\"
    - java -mx512m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
