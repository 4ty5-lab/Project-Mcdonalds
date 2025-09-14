import os
import pickle
import re
import numpy as np
# import spacy  # Commented out for now - having issues with installation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict
import string

# TODO: Get spacy working properly later
# For now using basic text processing
# try:
#     nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# except OSError:
#     import subprocess
#     import sys
#     subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
#     nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class TextProcessor:
    def __init__(self):
        # Basic stop words - started small, kept adding more
        self.stop_words = set([
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        ])
        self.punct_re = re.compile(r'[^\w\s]')
        # self.nlp = nlp  # Will add this back when spacy works

    def clean_text(self, text):
        if not text:
            return ""
        
        text = text.lower()
        # Remove punctuation
        text = self.punct_re.sub(' ', text)
        
        # Basic tokenization and filtering
        words = text.split()
        # Filter stop words and short words
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return ' '.join(filtered)

    # TODO: Add proper lemmatization when spacy is working
    def basic_stem(self, word):
        """Very basic stemming - just removes common suffixes"""
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word


class FileOrganizer:
    def __init__(self, model_dir="file_models"):
        self.model_dir = model_dir
        self.text_processor = TextProcessor()
        
        # Using default params for now - might tune later
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        self.classifier = GradientBoostingClassifier(n_estimators=50)  # Reduced for speed
        self.label_encoder = LabelEncoder()
        self.category_stats = defaultdict(int)
        
        # Common file extensions mapping - built this up over time
        self.ext_categories = {
            'txt': 'documents', 'doc': 'documents', 'docx': 'documents', 'pdf': 'documents',
            'jpg': 'images', 'jpeg': 'images', 'png': 'images', 'gif': 'images', 'bmp': 'images',
            'mp4': 'videos', 'avi': 'videos', 'mov': 'videos', 'mkv': 'videos',
            'mp3': 'audio', 'wav': 'audio', 'flac': 'audio',
            'zip': 'archives', 'rar': 'archives', '7z': 'archives', 'tar': 'archives',
            'py': 'code', 'js': 'code', 'html': 'code', 'css': 'code', 'java': 'code',
            'exe': 'executables', 'msi': 'executables', 'app': 'executables'
        }
        
        self._setup()

    def _setup(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self._load_saved_models()

    def _load_saved_models(self):
        """Load previously saved models if they exist"""
        try:
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open(os.path.join(self.model_dir, 'stats.pkl'), 'rb') as f:
                self.category_stats = pickle.load(f)
            print("Loaded existing models")
        except (FileNotFoundError, EOFError) as e:
            print("No existing models found, will create new ones")
            pass

    def _save_models(self):
        """Save trained models"""
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
            if ext in ['txt', 'md', 'py', 'js', 'html', 'css'] and size < 1000000:  # 1MB limit
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # First 1000 chars
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
        """Train the classifier with file paths and their categories"""
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
            
            # Calculate accuracy on training data (not ideal but simple)
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
                    
                    import shutil
                    shutil.move(file_path, dest_path)
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to move {file_path}: {e}")
        
        print(f"\nMoved {moved_count} files successfully")


class WebManager:
    def __init__(self, data_file="web_data.pkl"):
        self.data_file = data_file
        self.text_processor = TextProcessor()
        self.tabs = {}
        self.favorites = {}
        self.tab_id_counter = 0
        self._load_data()

    def _load_data(self):
        """Load saved data if it exists"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.tabs = data.get('tabs', {})
                    self.favorites = data.get('favorites', {})
                    self.tab_id_counter = data.get('tab_id_counter', 0)
                print(f"Loaded {len(self.tabs)} tabs and {len(self.favorites)} favorites")
            except Exception as e:
                print(f"Error loading data: {e}")
                self.tabs = {}
                self.favorites = {}
                self.tab_id_counter = 0

    def _save_data(self):
        """Save current data"""
        try:
            data = {
                'tabs': self.tabs,
                'favorites': self.favorites,
                'tab_id_counter': self.tab_id_counter
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving data: {e}")

    def add_tab(self, url, title, content=""):
        """Add a new tab"""
        self.tab_id_counter += 1
        tab_id = self.tab_id_counter
        
        # Create URL hash for deduplication
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
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
        print(f"Added tab: {title}")
        return tab_id

    def update_tab_activity(self, tab_id):
        """Update last activity time for a tab"""
        if tab_id in self.tabs:
            self.tabs[tab_id]['last_active'] = datetime.now()
            self.tabs[tab_id]['visit_count'] += 1
            self._save_data()
            return True
        return False

    def find_inactive_tabs(self, threshold_minutes=30):
        """Find tabs that haven't been active recently"""
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
        """Search through open tabs"""
        if not query.strip():
            return []
        
        clean_query = self.text_processor.clean_text(query).lower()
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
        """Add a bookmark to favorites"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        self.favorites[url_hash] = {
            'url': url,
            'title': title,
            'category': category,
            'added': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 1
        }
        
        self._save_data()
        print(f"Added favorite: {title}")
        return True

    def remove_favorite(self, url):
        """Remove a bookmark"""
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
        favorites = list(self.favorites.values())
        if category:
            favorites = [f for f in favorites if f['category'] == category]
        
        # Sort by access count and recency
        favorites.sort(key=lambda x: (x['access_count'], x['last_accessed']), reverse=True)
        return favorites

    def search_favorites(self, query, category=None):
        """Search through bookmarks"""
        if not query.strip():
            return self.get_favorites(category)
        
        clean_query = self.text_processor.clean_text(query).lower()
        query_words = set(clean_query.split())
        
        favorites = self.get_favorites(category)
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
        """Record access to a favorite"""
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


# Simple demo/test function
def demo():
    print("=== File Organizer Demo ===")
    organizer = FileOrganizer()
    
    # Create some test files
    test_files = []
    test_categories = []
    
    # Create test directory if it doesn't exist
    test_dir = "test_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Create some sample files
    sample_files = [
        ("document.txt", "documents", "This is a text document with some content."),
        ("image.jpg", "images", ""),  # Binary file, no content
        ("script.py", "code", "print('Hello World')\n# This is a Python script"),
        ("data.csv", "documents", "name,age,city\nJohn,25,NYC\nJane,30,LA"),
        ("archive.zip", "archives", ""),  # Binary file
    ]
    
    for filename, category, content in sample_files:
        filepath = os.path.join(test_dir, filename)
        if content:  # Text file
            with open(filepath, 'w') as f:
                f.write(content)
        else:  # Create empty file for binary types
            with open(filepath, 'w') as f:
                f.write("")
        
        test_files.append(filepath)
        test_categories.append(category)
    
    # Train the organizer
    print("Training file organizer...")
    score = organizer.train_categorizer(test_files, test_categories)
    print(f"Training score: {score:.2f}")
    
    # Test classification
    print("\nTesting classification...")
    predictions = organizer.categorize_files(test_files)
    for file, actual, predicted in zip(test_files, test_categories, predictions):
        status = "✓" if actual == predicted else "✗"
        print(f"{status} {os.path.basename(file)}: {actual} -> {predicted}")
    
    print("\n=== Web Manager Demo ===")
    web_manager = WebManager()
    
    # Add some test tabs
    web_manager.add_tab("https://python.org", "Python Programming Language")
    web_manager.add_tab("https://github.com", "GitHub - Code Hosting")
    web_manager.add_tab("https://stackoverflow.com", "Stack Overflow - Programming Q&A")
    
    # Add some favorites
    web_manager.add_favorite("https://python.org", "Python Official Site", "programming")
    web_manager.add_favorite("https://github.com", "GitHub", "development")
    web_manager.add_favorite("https://docs.python.org", "Python Documentation", "programming")
    
    # Search tests
    print("\nSearching tabs for 'python':")
    results = web_manager.search_tabs("python")
    for result in results:
        print(f"- {result['title']} (score: {result['score']})")
    
    print("\nSearching favorites for 'programming':")
    results = web_manager.search_favorites("programming")
    for result in results:
        print(f"- {result['title']} ({result['category']})")
    
    # Show stats
    print(f"\nWeb Manager Stats: {web_manager.get_stats()}")
    print(f"File Organizer Stats: {organizer.get_category_stats()}")


if __name__ == "__main__":
    demo()
