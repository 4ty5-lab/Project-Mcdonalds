#!/usr/bin/env python3
"""
NLP File Organizer & Web Manager - Smart organization with ML

Complete implementation with all features from both versions
"""

import os
import pickle
import re
import numpy as np
import hashlib
import argparse
import shutil
from collections import defaultdict
from datetime import datetime, timedelta

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Try to import spacy, but have fallback
try:
    import spacy
    SPACY_AVAILABLE = True
    print("spaCy is available. Using advanced NLP features.")
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Using basic text processing.")

# Shortcuts
dt = datetime
dd = defaultdict
td = timedelta

class TextProcessor:
    """Cleans text for ML processing with fallback options"""
    
    def __init__(self, spacy_available=False):  # Add parameter with default
        # Basic stop words
        self.stop_words = set([
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        ])
        self.punct_re = re.compile(r'[^\w\s]')
        
        # Initialize spaCy if available
        self.nlp = None
        if spacy_available:  # Use the parameter instead of global variable
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                self.nlp.max_length = 1000000  # handle big files
                print("spaCy model loaded successfully.")
            except OSError:
                print("spaCy model not found. Using basic text processing.")
                spacy_available = False  # Update local variable

    def clean_text(self, text):
        """Clean and process text with spaCy if available, otherwise use basic processing"""
        if not text:
            return ""
        
        text = text.lower()
        # Remove punctuation
        text = self.punct_re.sub(' ', text)
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                # Filter out stop words and single chars, use lemmas
                return ' '.join([t.lemma_ for t in doc if t.lemma_ not in self.stop_words and len(t.lemma_) > 1])
            except Exception as e:
                print(f"spaCy processing failed: {e}. Falling back to basic processing.")
        
        # Basic tokenization and filtering as fallback
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        return ' '.join(filtered)


class FileOrganizer:
    """Main file organization class using ML"""
    
    def __init__(self, model_dir=None):  # Make parameter optional
        if model_dir is None:
            # Use a directory in user's home or documents folder
            model_dir = os.path.join(os.path.expanduser("~"), "nlp_file_organizer_models")
        
        self.model_dir = model_dir
        self.text_processor = TextProcessor(SPACY_AVAILABLE)
        
        # ML components
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        self.classifier = GradientBoostingClassifier(n_estimators=100, verbose=0)
        self.label_encoder = LabelEncoder()
        self.category_stats = defaultdict(int)
        
        # Common file extensions mapping
        self.ext_categories = {
            'txt': 'documents', 'doc': 'documents', 'docx': 'documents', 'pdf': 'documents', 'rtf': 'documents',
            'jpg': 'images', 'jpeg': 'images', 'png': 'images', 'gif': 'images', 'bmp': 'images', 'svg': 'images',
            'mp4': 'videos', 'avi': 'videos', 'mov': 'videos', 'mkv': 'videos', 'flv': 'videos', 'wmv': 'videos',
            'mp3': 'audio', 'wav': 'audio', 'flac': 'audio', 'aac': 'audio', 'ogg': 'audio',
            'zip': 'archives', 'rar': 'archives', '7z': 'archives', 'tar': 'archives', 'gz': 'archives',
            'py': 'code', 'js': 'code', 'html': 'code', 'css': 'code', 'java': 'code', 'cpp': 'code', 'c': 'code',
            'exe': 'executables', 'msi': 'executables', 'app': 'executables', 'dmg': 'executables',
            'xls': 'spreadsheets', 'xlsx': 'spreadsheets', 'csv': 'spreadsheets',
            'ppt': 'presentations', 'pptx': 'presentations'
        }
        
        self._setup()

    def _setup(self):
        """Initialize directories and load saved models"""
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_saved_models()

    def _load_saved_models(self):
        """Load previously trained models if they exist"""
        try:
            # Load vectorizer
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            # Load classifier
            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            # Load label encoder
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            # Load stats
            with open(os.path.join(self.model_dir, 'stats.pkl'), 'rb') as f:
                self.category_stats = pickle.load(f)
            print("Loaded existing models")
        except (FileNotFoundError, EOFError, pickle.PickleError) as e:
            print("No existing models found or error loading, will create new ones")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            with open(os.path.join(self.model_dir, 'stats.pkl'), 'wb') as f:
                pickle.dump(self.category_stats, f)
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")

    def _get_file_features(self, file_path):
        """Extract features from file for classification"""
        try:
            if not os.path.exists(file_path):
                return ""
            
            name = os.path.basename(file_path)
            ext = os.path.splitext(name)[1].lstrip('.').lower()
            folder = os.path.basename(os.path.dirname(file_path))
            
            # Get file size category
            try:
                size = os.path.getsize(file_path)
                if size < 1024:
                    size_cat = "tiny"
                elif size < 1024*1024:
                    size_cat = "small"
                elif size < 100*1024*1024:
                    size_cat = "medium"
                else:
                    size_cat = "large"
            except:
                size_cat = "unknown"
            
            features = [name, ext, folder, size_cat]
            
            # Try to read text content for certain file types
            text_extensions = ['txt', 'md', 'py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'csv', 'json', 'xml']
            if ext in text_extensions and size < 5000000:  # 5MB limit for text files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(2000)  # First 2000 chars
                        features.append(content)
                except:
                    pass  # If can't read, just skip content
                    
            return self.text_processor.clean_text(' '.join(features))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""

    def guess_category_by_extension(self, file_path):
        """Fallback method using file extension"""
        ext = os.path.splitext(file_path)[1].lstrip('.').lower()
        return self.ext_categories.get(ext, 'other')

    def train_categorizer(self, file_paths, categories):
        """Train the ML model on file examples"""
        if len(file_paths) != len(categories):
            print("Error: file_paths and categories must have same length")
            return 0.0
            
        if len(set(categories)) < 2:
            print("Error: Need at least 2 different categories")
            return 0.0
        
        print(f"Training on {len(file_paths)} files...")
        
        # Extract features
        X_text = []
        valid_indices = []
        for i, fp in enumerate(file_paths):
            features = self._get_file_features(fp)
            if features:  # Only include if we got valid features
                X_text.append(features)
                valid_indices.append(i)
        
        if len(X_text) == 0:
            print("No valid features extracted")
            return 0.0
        
        # Filter categories to match valid features
        valid_categories = [categories[i] for i in valid_indices]
        
        try:
            # Vectorize text features
            X = self.vectorizer.fit_transform(X_text)
            
            # Encode labels
            y = self.label_encoder.fit_transform(valid_categories)
            
            # Train classifier
            self.classifier.fit(X, y)
            
            # Calculate accuracy on training data
            score = self.classifier.score(X, y)
            
            # Update stats
            for cat in valid_categories:
                self.category_stats[cat] += 1
                
            self._save_models()
            print(f"Training completed. Accuracy: {score:.2f}")
            return score
            
        except Exception as e:
            print(f"Training failed: {e}")
            return 0.0

    def categorize_files(self, file_paths):
        """Categorize a list of files"""
        if not hasattr(self.classifier, 'classes_'):
            print("Classifier not trained yet. Using extension-based fallback.")
            return [self.guess_category_by_extension(fp) for fp in file_paths]
        
        try:
            X_text = [self._get_file_features(fp) for fp in file_paths]
            
            # Handle empty features
            results = []
            for i, features in enumerate(X_text):
                if not features:
                    # Fallback to extension-based classification
                    results.append(self.guess_category_by_extension(file_paths[i]))
                else:
                    try:
                        X = self.vectorizer.transform([features])
                        pred = self.classifier.predict(X)
                        category = self.label_encoder.inverse_transform(pred)[0]
                        results.append(category)
                    except:
                        # Another fallback
                        results.append(self.guess_category_by_extension(file_paths[i]))
            
            return results
            
        except Exception as e:
            print(f"Classification failed: {e}")
            # Ultimate fallback
            return [self.guess_category_by_extension(fp) for fp in file_paths]

    def get_category_stats(self):
        """Return category usage statistics"""
        return dict(self.category_stats)

    def organize_files(self, source_dir, target_dir, dry_run=True):
        """Organize files from source to target directory by category"""
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} doesn't exist")
            return
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Get all files in source directory
        files = []
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isfile(item_path):
                files.append(item_path)
        
        if not files:
            print("No files found in source directory")
            return
        
        print(f"Found {len(files)} files to organize")
        
        # Categorize files
        categories = self.categorize_files(files)
        
        # Group by category
        moves = defaultdict(list)
        for file_path, category in zip(files, categories):
            moves[category].append(file_path)
        
        # Show what would be moved
        for category, file_list in moves.items():
            print(f"\n{category.upper()} ({len(file_list)} files):")
            for f in file_list[:5]:  # Show first 5
                print(f"  - {os.path.basename(f)}")
            if len(file_list) > 5:
                print(f"  ... and {len(file_list) - 5} more")
        
        if dry_run:
            print("\nDRY RUN - no files were actually moved")
            return
        
        # Actually move files
        moved_count = 0
        for category, file_list in moves.items():
            category_dir = os.path.join(target_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            for file_path in file_list:
                try:
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(category_dir, filename)
                    
                    # Handle name conflicts
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(filename)
                        dest_path = os.path.join(category_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(file_path, dest_path)
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to move {file_path}: {e}")
        
        print(f"\nMoved {moved_count} files successfully")


class WebManager:
    """Manages browser tabs and favorites with NLP search"""
    
    def __init__(self, data_file="web_data.pkl"):
        self.data_file = data_file
        self.text_processor = TextProcessor(SPACY_AVAILABLE)  # Pass the global variable
        self.tabs = {}  # active tabs
        self.favorites = {}  # bookmarks
        self.tab_id_counter = 0
        self._load_data()

    def _load_data(self):
        """Load saved web data"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.tabs = data.get('tabs', {})
                    self.favorites = data.get('favorites', {})
                    self.tab_id_counter = data.get('tab_id_counter', 0)
                print(f"Loaded {len(self.tabs)} tabs and {len(self.favorites)} favorites")
            except Exception as e:
                print(f"Error loading web data: {e}")
                self.tabs = {}
                self.favorites = {}
                self.tab_id_counter = 0

    def _save_data(self):
        """Save web data to disk"""
        try:
            data = {
                'tabs': self.tabs,
                'favorites': self.favorites,
                'tab_id_counter': self.tab_id_counter
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving web data: {e}")

    def add_tab(self, url, title, content=""):
        """Add a new browser tab"""
        self.tab_id_counter += 1
        tab_id = self.tab_id_counter
        
        # Create URL hash for deduplication
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        processed_title = self.text_processor.clean_text(title)
        
        self.tabs[tab_id] = {
            'url': url,
            'title': title,
            'clean_title': processed_title,
            'url_hash': url_hash,
            'created': datetime.now(),
            'last_active': datetime.now(),
            'visit_count': 1
        }
        
        self._save_data()
        print(f"Added tab {tab_id}: {title}")
        return tab_id

    def update_tab_activity(self, tab_id):
        """Update tab activity timestamp"""
        if tab_id in self.tabs:
            self.tabs[tab_id]['last_active'] = datetime.now()
            self.tabs[tab_id]['visit_count'] += 1
            self._save_data()
            return True
        return False

    def find_inactive_tabs(self, threshold_minutes=30):
        """Find tabs that haven't been used recently"""
        threshold = timedelta(minutes=threshold_minutes)
        now = datetime.now()
        
        inactive = []
        for tab_id, tab in self.tabs.items():
            inactive_time = now - tab['last_active']
            if inactive_time > threshold:
                inactive.append({
                    'id': tab_id,
                    'title': tab['title'],
                    'url': tab['url'],
                    'inactive_minutes': inactive_time.total_seconds() / 60
                })
        
        return sorted(inactive, key=lambda x: x['inactive_minutes'], reverse=True)

    def search_tabs(self, query, limit=10):
        """Search tabs using NLP similarity"""
        if not query or not self.tabs:
            return []
            
        clean_query = self.text_processor.clean_text(query)
        
        # Use spaCy similarity if available
        if SPACY_AVAILABLE and self.text_processor.nlp:
            try:
                query_doc = self.text_processor.nlp(clean_query)
                results = []
                
                for tab_id, tab in self.tabs.items():
                    # Calculate similarity between query and tab title
                    tab_doc = self.text_processor.nlp(tab['clean_title'])
                    sim = query_doc.similarity(tab_doc)
                    if sim > 0.3:  # similarity threshold
                        results.append({
                            'id': tab_id,
                            'title': tab['title'],
                            'url': tab['url'],
                            'score': sim
                        })
                
                # Sort by similarity score
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:limit]
            except Exception as e:
                print(f"spaCy similarity failed: {e}. Falling back to keyword search.")
        
        # Fallback to keyword matching
        query_words = set(clean_query.split())
        results = []
        
        for tab_id, tab in self.tabs.items():
            # Simple keyword matching
            title_words = set(tab['clean_title'].split())
            url_words = set(self.text_processor.clean_text(tab['url']).split())
            
            # Calculate simple similarity score
            title_matches = len(query_words & title_words)
            url_matches = len(query_words & url_words)
            
            score = title_matches * 2 + url_matches  # Weight title matches more
            
            if score > 0:
                results.append({
                    'id': tab_id,
                    'title': tab['title'],
                    'url': tab['url'],
                    'score': score
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def close_tab(self, tab_id):
        """Close a tab"""
        if tab_id in self.tabs:
            title = self.tabs[tab_id]['title']
            del self.tabs[tab_id]
            self._save_data()
            print(f"Closed tab: {title}")
            return True
        return False

    def add_favorite(self, url, title, category="general"):
        """Add URL to favorites"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
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
        print(f"Added favorite: {title} to {category}")
        return True

    def remove_favorite(self, url):
        """Remove URL from favorites"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.favorites:
            title = self.favorites[url_hash]['title']
            del self.favorites[url_hash]
            self._save_data()
            print(f"Removed favorite: {title}")
            return True
        return False

    def get_favorites(self, category=None):
        """Get favorites, optionally filtered by category"""
        if category:
            return [fav for fav in self.favorites.values() if fav['category'] == category]
        
        # Sort by access count and recency
        favorites = list(self.favorites.values())
        favorites.sort(key=lambda x: (x['access_count'], x['last_accessed']), reverse=True)
        return favorites

    def search_favorites(self, query, category=None):
        """Search favorites using NLP"""
        if not query or not self.favorites:
            return self.get_favorites(category)
            
        clean_query = self.text_processor.clean_text(query)
        favorites = self.get_favorites(category)
        
        # Use spaCy similarity if available
        if SPACY_AVAILABLE and self.text_processor.nlp:
            try:
                query_doc = self.text_processor.nlp(clean_query)
                results = []
                
                for fav in favorites:
                    # Search in title, URL, and category
                    text = f"{fav['title']} {fav['url']} {fav['category']}"
                    text_doc = self.text_processor.nlp(text)
                    sim = query_doc.similarity(text_doc)
                    if sim > 0.2:  # lower threshold for favorites
                        results.append({
                            'title': fav['title'],
                            'url': fav['url'],
                            'category': fav['category'],
                            'score': sim
                        })
                
                # Sort by similarity score
                results.sort(key=lambda x: x['score'], reverse=True)
                return results
            except Exception as e:
                print(f"spaCy similarity failed: {e}. Falling back to keyword search.")
        
        # Fallback to keyword matching
        query_words = set(clean_query.split())
        results = []
        
        for fav in favorites:
            # Search in title, URL, and category
            search_text = f"{fav['title']} {fav['url']} {fav['category']}"
            clean_text = self.text_processor.clean_text(search_text)
            text_words = set(clean_text.split())
            
            matches = len(query_words & text_words)
            if matches > 0:
                results.append({
                    'title': fav['title'],
                    'url': fav['url'],
                    'category': fav['category'],
                    'score': matches
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def access_favorite(self, url):
        """Record favorite access"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.favorites:
            fav = self.favorites[url_hash]
            fav['last_accessed'] = datetime.now()
            fav['access_count'] += 1
            self._save_data()
            return fav
        return None

    def get_stats(self):
        """Get usage statistics"""
        total_tabs = len(self.tabs)
        total_favorites = len(self.favorites)
        
        if self.tabs:
            avg_visits = sum(tab['visit_count'] for tab in self.tabs.values()) / total_tabs
        else:
            avg_visits = 0
        
        categories = {}
        for fav in self.favorites.values():
            cat = fav['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_tabs': total_tabs,
            'total_favorites': total_favorites,
            'avg_tab_visits': avg_visits,
            'favorite_categories': categories
        }


# CLI Commands and shortcuts
def cmd_organize(organizer, source_dir, target_dir=None, dry_run=True):
    """Organize files in directory"""
    if target_dir is None:
        target_dir = source_dir + "_organized"
    
    organizer.organize_files(source_dir, target_dir, dry_run)

def cmd_train(organizer, training_dir, category_map=None):
    """Train the organizer with files in directory"""
    if not os.path.exists(training_dir):
        print(f"Training directory {training_dir} doesn't exist")
        return
    
    file_paths = []
    categories = []
    
    # If category_map is provided, use it to map subdirectories to categories
    if category_map:
        for category, patterns in category_map.items():
            for pattern in patterns:
                pattern_path = os.path.join(training_dir, pattern)
                if os.path.exists(pattern_path):
                    for item in os.listdir(pattern_path):
                        item_path = os.path.join(pattern_path, item)
                        if os.path.isfile(item_path):
                            file_paths.append(item_path)
                            categories.append(category)
    else:
        # Use subdirectory names as categories
        for category in os.listdir(training_dir):
            category_path = os.path.join(training_dir, category)
            if os.path.isdir(category_path):
                for item in os.listdir(category_path):
                    item_path = os.path.join(category_path, item)
                    if os.path.isfile(item_path):
                        file_paths.append(item_path)
                        categories.append(category)
    
    if not file_paths:
        print("No training files found")
        return
    
    print(f"Training with {len(file_paths)} files...")
    score = organizer.train_categorizer(file_paths, categories)
    print(f"Training completed with accuracy: {score:.2f}")

def cmd_search_tabs(web_manager, query, limit=5):
    """Search browser tabs"""
    results = web_manager.search_tabs(query, limit)
    if not results:
        print("No matching tabs found")
        return
    
    print(f"Found {len(results)} matching tabs:")
    for result in results:
        print(f"[{result['id']}] {result['title']} - {result['url']} (score: {result['score']:.2f})")

def cmd_list_tabs(web_manager):
    """List all open tabs"""
    if not web_manager.tabs:
        print("No tabs open")
        return
    
    print("Open tabs:")
    for tab_id, tab in web_manager.tabs.items():
        inactive_time = datetime.now() - tab['last_active']
        inactive_mins = inactive_time.total_seconds() / 60
        print(f"[{tab_id}] {tab['title']} (inactive: {inactive_mins:.1f}m, visits: {tab['visit_count']})")

def cmd_list_inactive_tabs(web_manager, threshold_minutes=30):
    """List inactive tabs"""
    inactive = web_manager.find_inactive_tabs(threshold_minutes)
    if not inactive:
        print(f"No tabs inactive for more than {threshold_minutes} minutes")
        return
    
    print(f"Inactive tabs (> {threshold_minutes} minutes):")
    for tab in inactive:
        print(f"[{tab['id']}] {tab['title']} - inactive for {tab['inactive_minutes']:.1f} minutes")

def cmd_close_tab(web_manager, tab_id):
    """Close a tab"""
    if web_manager.close_tab(tab_id):
        print(f"Tab {tab_id} closed")
    else:
        print(f"Tab {tab_id} not found")

def cmd_add_fav(web_manager, url, title, category="general"):
    """Add favorite bookmark"""
    web_manager.add_favorite(url, title, category)
    print(f"Added {title} to {category}")

def cmd_remove_fav(web_manager, url):
    """Remove favorite bookmark"""
    if web_manager.remove_favorite(url):
        print(f"Favorite removed")
    else:
        print(f"Favorite not found")

def cmd_search_favs(web_manager, query, category=None):
    """Search favorites"""
    results = web_manager.search_favorites(query, category)
    if not results:
        print("No matching favorites found")
        return
    
    print(f"Found {len(results)} matching favorites:")
    for result in results:
        print(f"{result['title']} ({result['category']}) - {result['url']} (score: {result['score']:.2f})")

def cmd_list_favs(web_manager, category=None):
    """List favorites"""
    favorites = web_manager.get_favorites(category)
    if not favorites:
        if category:
            print(f"No favorites in category '{category}'")
        else:
            print("No favorites saved")
        return
    
    if category:
        print(f"Favorites in category '{category}':")
    else:
        print("All favorites:")
    
    for fav in favorites:
        last_access = (datetime.now() - fav['last_accessed']).total_seconds() / 3600
        print(f"{fav['title']} - {fav['url']} (accessed {last_access:.1f}h ago, {fav['access_count']} times)")

def cmd_stats(organizer, web_manager):
    """Show statistics"""
    print("=== FILE ORGANIZER STATS ===")
    file_stats = organizer.get_category_stats()
    for category, count in file_stats.items():
        print(f"{category}: {count} files")
    
    print("\n=== WEB MANAGER STATS ===")
    web_stats = web_manager.get_stats()
    print(f"Total tabs: {web_stats['total_tabs']}")
    print(f"Total favorites: {web_stats['total_favorites']}")
    print(f"Average tab visits: {web_stats['avg_tab_visits']:.1f}")
    print("Favorite categories:")
    for category, count in web_stats['favorite_categories'].items():
        print(f"  {category}: {count}")

# Quick shortcuts
org = cmd_organize
train = cmd_train
st = cmd_search_tabs
lt = cmd_list_tabs
lit = cmd_list_inactive_tabs
ct = cmd_close_tab
af = cmd_add_fav
rf = cmd_remove_fav
sf = cmd_search_favs
lf = cmd_list_favs
stats = cmd_stats


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="NLP File Organizer and Web Manager")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # File organization commands
    org_parser = subparsers.add_parser('organize', help='Organize files in directory')
    org_parser.add_argument('source_dir', help='Source directory to organize')
    org_parser.add_argument('--target-dir', help='Target directory (default: source_organized)')
    org_parser.add_argument('--no-dry-run', action='store_true', help='Actually move files (not dry run)')
    
    train_parser = subparsers.add_parser('train', help='Train the file organizer')
    train_parser.add_argument('training_dir', help='Directory with categorized training files')
    
    # Web manager commands
    search_tabs_parser = subparsers.add_parser('search-tabs', help='Search browser tabs')
    search_tabs_parser.add_argument('query', help='Search query')
    search_tabs_parser.add_argument('--limit', type=int, default=5, help='Maximum results to show')
    
    list_tabs_parser = subparsers.add_parser('list-tabs', help='List all open tabs')
    
    inactive_tabs_parser = subparsers.add_parser('inactive-tabs', help='List inactive tabs')
    inactive_tabs_parser.add_argument('--minutes', type=int, default=30, help='Inactivity threshold in minutes')
    
    close_tab_parser = subparsers.add_parser('close-tab', help='Close a tab')
    close_tab_parser.add_argument('tab_id', type=int, help='Tab ID to close')
    
    add_fav_parser = subparsers.add_parser('add-favorite', help='Add a favorite')
    add_fav_parser.add_argument('url', help='URL to add')
    add_fav_parser.add_argument('title', help='Title for the favorite')
    add_fav_parser.add_argument('--category', default='general', help='Category for the favorite')
    
    remove_fav_parser = subparsers.add_parser('remove-favorite', help='Remove a favorite')
    remove_fav_parser.add_argument('url', help='URL to remove')
    
    search_favs_parser = subparsers.add_parser('search-favorites', help='Search favorites')
    search_favs_parser.add_argument('query', help='Search query')
    search_favs_parser.add_argument('--category', help='Filter by category')
    
    list_favs_parser = subparsers.add_parser('list-favorites', help='List favorites')
    list_favs_parser.add_argument('--category', help='Filter by category')
    
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    args = parser.parse_args()
    
    # Initialize components
    organizer = FileOrganizer()
    web_manager = WebManager()
    
    if args.command == 'organize':
        cmd_organize(organizer, args.source_dir, args.target_dir, not args.no_dry_run)
    elif args.command == 'train':
        cmd_train(organizer, args.training_dir)
    elif args.command == 'search-tabs':
        cmd_search_tabs(web_manager, args.query, args.limit)
    elif args.command == 'list-tabs':
        cmd_list_tabs(web_manager)
    elif args.command == 'inactive-tabs':
        cmd_list_inactive_tabs(web_manager, args.minutes)
    elif args.command == 'close-tab':
        cmd_close_tab(web_manager, args.tab_id)
    elif args.command == 'add-favorite':
        cmd_add_fav(web_manager, args.url, args.title, args.category)
    elif args.command == 'remove-favorite':
        cmd_remove_fav(web_manager, args.url)
    elif args.command == 'search-favorites':
        cmd_search_favs(web_manager, args.query, args.category)
    elif args.command == 'list-favorites':
        cmd_list_favs(web_manager, args.category)
    elif args.command == 'stats':
        cmd_stats(organizer, web_manager)
    elif args.command == 'demo':
        run_demo(organizer, web_manager)
    else:
        parser.print_help()


def run_demo(organizer, web_manager):
    """Run a demonstration of the system"""
    print("=== NLP FILE ORGANIZER & WEB MANAGER DEMO ===")
    
    # Create test files for demonstration
    test_dir = "test_demo_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
        # Create some sample files in categories
        categories = {
            'documents': ['report.txt', 'notes.md', 'essay.doc'],
            'images': ['photo.jpg', 'screenshot.png', 'diagram.bmp'],
            'code': ['script.py', 'program.js', 'styles.css']
        }
        
        for category, files in categories.items():
            category_dir = os.path.join(test_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for file in files:
                file_path = os.path.join(category_dir, file)
                with open(file_path, 'w') as f:
                    if category == 'documents':
                        f.write("This is a sample text document for demonstration.")
                    elif category == 'code':
                        if file.endswith('.py'):
                            f.write("print('Hello, World!')\n# This is a Python script")
                        elif file.endswith('.js'):
                            f.write("console.log('Hello, World!');\n// This is a JavaScript file")
                        else:
                            f.write("body { color: black; }\n/* This is a CSS file */")
                    else:
                        f.write("")  # Empty file for images
    
    print("1. Training file organizer with demo files...")
    cmd_train(organizer, test_dir)
    
    print("\n2. Adding some web tabs and favorites...")
    web_manager.add_tab("https://python.org", "Python Programming Language")
    web_manager.add_tab("https://github.com", "GitHub - Code Hosting")
    web_manager.add_tab("https://stackoverflow.com", "Stack Overflow - Programming Q&A")
    
    web_manager.add_favorite("https://python.org", "Python Official Site", "programming")
    web_manager.add_favorite("https://github.com", "GitHub", "development")
    web_manager.add_favorite("https://docs.python.org", "Python Documentation", "programming")
    
    print("\n3. Searching for 'python' in tabs...")
    cmd_search_tabs(web_manager, "python")
    
    print("\n4. Searching for 'programming' in favorites...")
    cmd_search_favs(web_manager, "programming")
    
    print("\n5. Showing statistics...")
    cmd_stats(organizer, web_manager)
    
    print("\n=== DEMO COMPLETE ===")
    print("You can now try other commands like:")
    print("  python nlp_organizer.py organize /path/to/your/files")
    print("  python nlp_organizer.py search-tabs 'your query'")
    print("  python nlp_organizer.py list-favorites")


if __name__ == "__main__":
    main()
    
    # If you read this, you're smart! 😊
    # This system combines machine learning with practical file organization
    # and web management features in a single, cohesive application.