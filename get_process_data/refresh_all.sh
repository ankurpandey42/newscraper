cd ~
python3 mongo_driver.py --kill articles
python3 webcrawler.py
python3 group_articles_by_flag.py
python3 lemmatize_articles.py
python3 NLP_machine.py
cd newscraper
