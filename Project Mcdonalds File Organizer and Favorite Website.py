import os
import pickle
import re
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class TextProcessor:
    __slots__ = ['stop_words', 'punct_re', 'nlp']
    
    def __init__(self):
        self.nlp = nlp
        self.stop_words = self.nlp.Defaults.stop_words
        self.punct_re = re.compile(r'[^\w\s]')
        self.nlp.max_length = 1000000

    def clean_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = self.punct_re.sub(' ', text)
        doc = self.nlp(text)
        return ' '.join([t.lemma_ for t in doc if t.lemma_ not in self.stop_words and len(t.lemma_) > 1])


class FileOrganizer:
    __slots__ = ['model_dir', 'text_processor', 'vectorizer', 'classifier', 'label_encoder', 'category_stats']
    
    def __init__(self, model_dir="file_models"):
        self.model_dir = model_dir
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
        self.classifier = GradientBoostingClassifier(n_estimators=100, verbose=0)
        self.label_encoder = LabelEncoder()
        self.category_stats = defaultdict(int)
        self._setup()

    def _setup(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_saved_models()

    def _load_saved_models(self):
        try:
            with open(os.path.join(self.model_dir, 'vec.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(self.model_dir, 'clf.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            with open(os.path.join(self.model_dir, 'le.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open(os.path.join(self.model_dir, 'stats.pkl'), 'rb') as f:
                self.category_stats = pickle.load(f)
        except (FileNotFoundError, EOFError):
            pass

    def _save_models(self):
        with open(os.path.join(self.model_dir, 'vec.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, 'clf.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(os.path.join(self.model_dir, 'le.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(os.path.join(self.model_dir, 'stats.pkl'), 'wb') as f:
            pickle.dump(self.category_stats, f)

    def _get_file_info(self, file_path):
        try:
            name = os.path.basename(file_path)
            ext = os.path.splitext(name)[1].lstrip('.').lower()
            folder = os.path.basename(os.path.dirname(file_path))
            size = os.path.getsize(file_path)
            
            features = [name, ext, folder, f"s{size//1024}"]
            
            if ext in ['txt', 'md'] and size < 1e6:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        features.append(f.read(500))
                except Exception:
                    pass
                    
            return self.text_processor.clean_text(' '.join(features))
        except Exception:
            return ""

    def train_categorizer(self, file_paths, categories):
        if len(set(categories)) < 2:
            return 0.0
            
        X_text = [self._get_file_info(fp) for fp in file_paths]
        X = self.vectorizer.fit_transform(X_text)
        y = self.label_encoder.fit_transform(categories)
        
        self.classifier.fit(X, y)
        score = self.classifier.score(X, y)
        
        for cat in categories:
            self.category_stats[cat] += 1
            
        self._save_models()
        return score

    def categorize_files(self, file_paths):
        try:
            X_text = [self._get_file_info(fp) for fp in file_paths]
            X = self.vectorizer.transform(X_text)
            return self.label_encoder.inverse_transform(self.classifier.predict(X))
        except Exception:
            return [None] * len(file_paths)

    def get_category_stats(self):
        return dict(self.category_stats)


class WebManager:
    __slots__ = ['data_file', 'text_processor', 'tabs', 'favorites', 'tab_id_map']
    
    def __init__(self, data_file="web_data.pkl"):
        self.data_file = data_file
        self.text_processor = TextProcessor()
        self.tabs = {}
        self.favorites = {}
        self.tab_id_map = {}
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.tabs = data.get('tabs', {})
                    self.favorites = data.get('favorites', {})
                    self.tab_id_map = {v['id']: k for k, v in self.tabs.items()}
            except (Exception, EOFError):
                self.tabs = {}
                self.favorites = {}
                self.tab_id_map = {}

    def _save_data(self):
        data = {'tabs': self.tabs, 'favorites': self.favorites}
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)

    def add_tab(self, tab_id, url, title, content=""):
        url_hash = hashlib.sha1(url.encode()).hexdigest()
        processed_title = self.text_processor.clean_text(title)
        
        self.tabs[url_hash] = {
            'id': tab_id,
            'url': url,
            'title': title,
            'clean_title': processed_title,
            'last_active': datetime.now(),
            'visits': 1,
            'load_time': datetime.now().timestamp()
        }
        self.tab_id_map[tab_id] = url_hash
        self._save_data()

    def update_activity(self, tab_id):
        if tab_id in self.tab_id_map:
            url_hash = self.tab_id_map[tab_id]
            tab = self.tabs[url_hash]
            tab['last_active'] = datetime.now()
            tab['visits'] += 1
            self._save_data()
            return True
        return False

    def find_inactive_tabs(self, threshold_min=15):
        threshold = timedelta(minutes=threshold_min)
        now = datetime.now()
        return [
            (tab['id'], (now - tab['last_active']).total_seconds() / 60)
            for tab in self.tabs.values()
            if now - tab['last_active'] > threshold
        ]

    def search_tabs(self, query, limit=10):
        if not query or not self.tabs:
            return []
            
        clean_query = self.text_processor.clean_text(query)
        query_doc = nlp(clean_query)
        results = []
        
        for tab in self.tabs.values():
            sim = query_doc.similarity(nlp(tab['clean_title']))
            if sim > 0.3:
                results.append((tab['id'], tab['title'], tab['url'], sim))
                
        return sorted(results, key=lambda x: x[3], reverse=True)[:limit]

    def add_favorite(self, url, title, category="general"):
        url_hash = hashlib.sha1(url.encode()).hexdigest()
        now = datetime.now()
        
        self.favorites[url_hash] = {
            'url': url,
            'title': title,
            'category': category,
            'added': now,
            'last_accessed': now,
            'access_count': 1
        }
        self._save_data()
        return True

    def remove_favorite(self, url):
        url_hash = hashlib.sha1(url.encode()).hexdigest()
        if url_hash in self.favorites:
            del self.favorites[url_hash]
            self._save_data()
            return True
        return False

    def get_favorites(self, category=None):
        if category:
            return [fav for fav in self.favorites.values() if fav['category'] == category]
        return list(self.favorites.values())

    def search_favorites(self, query, category=None):
        if not query or not self.favorites:
            return []
            
        clean_query = self.text_processor.clean_text(query)
        query_doc = nlp(clean_query)
        results = []
        
        favorites = self.get_favorites(category)
        for fav in favorites:
            text = fav['title'] + ' ' + fav['url'] + ' ' + fav['category']
            sim = query_doc.similarity(nlp(text))
            if sim > 0.2:
                results.append((fav['title'], fav['url'], fav['category'], sim))
                
        return sorted(results, key=lambda x: x[3], reverse=True)

    def access_favorite(self, url):
        url_hash = hashlib.sha1(url.encode()).hexdigest()
        if url_hash in self.favorites:
            fav = self.favorites[url_hash]
            fav['last_accessed'] = datetime.now()
            fav['access_count'] += 1
            self._save_data()
            return fav
        return None


if __name__ == "__main__":
    organizer = FileOrganizer()
    web_manager = WebManager()
    
    web_manager.add_tab(1, "https://python.org", "Python Programming Language")
    web_manager.add_tab(2, "https://github.com", "GitHub - Code Hosting")
    
    web_manager.add_favorite("https://python.org", "Python Official Site", "programming")
    web_manager.add_favorite("https://github.com", "GitHub", "development")
    web_manager.add_favorite("https://stackoverflow.com", "Stack Overflow", "programming")
    
    print("Search results for 'programming':")
    for result in web_manager.search_favorites("programming"):
        print(f"- {result[0]} ({result[1]})")
        
    print("\nFile organizer stats:", organizer.get_category_stats())
    
