"""
Full-functioning POS (Part-of-Speech) Tagger
Loads sentences from JSON and performs comprehensive linguistic analysis
"""

import json
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK resources
required_resources = [
    'punkt',
    'averaged_perceptron_tagger',
    'wordnet',
    'maxent_ne_chunker',
    'words',
    'universal_tagset'
]

# Download each resource; use a simple find/download to avoid probing non-zip files
for resource in required_resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)


class POSTagger:
    """
    A comprehensive POS tagger for analyzing sentences with multiple linguistic features
    """
    
    def __init__(self, json_file: str = 'outputs/intermediate_sentences.json'):
        """
        Initialize the POS tagger and load sentences from JSON
        
        Args:
            json_file: Path to the JSON file containing sentences
        """
        self.json_file = json_file
        self.sentences_data = []
        self.lemmatizer = WordNetLemmatizer()
        self.pos_results = []
        
    def load_sentences(self) -> bool:
        """
        Load sentences from the JSON file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(self.json_file)
            
            if not file_path.exists():
                print(f"Error: File '{self.json_file}' not found")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                self.sentences_data = json.load(f)
            
            print(f"✓ Loaded {len(self.sentences_data)} sentences from '{self.json_file}'")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def get_wordnet_pos(self, nltk_tag: str) -> str:
        """
        Convert NLTK POS tags to WordNet POS tags
        
        Args:
            nltk_tag: NLTK POS tag
            
        Returns:
            WordNet POS tag
        """
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        return None
    
    def analyze_sentence(self, sentence: str, target_lemma: str) -> Dict[str, Any]:
        """
        Modified to perform Thesis-Specific analysis (Section 3.2.1):
        1. Lemmatizes tokens FIRST to catch inflected forms (ran -> run).
        2. Checks if the lemma matches AND if it is a VERB.
        3. Extracts Context Window for the AI Judge.
        """
        # Tokenize and Tag
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        
        target_is_verb = False
        context_window = []
        lemmatized_tokens = []
        
        # 1. LEMMATIZATION LOOP
        for i, (token, tag) in enumerate(pos_tags):
            # Convert NLTK tag to WordNet tag for accurate lemmatization
            wn_pos = self.get_wordnet_pos(tag)
            if wn_pos:
                lemma = self.lemmatizer.lemmatize(token.lower(), pos=wn_pos)
            else:
                lemma = self.lemmatizer.lemmatize(token.lower())
            
            # Store for output
            lemmatized_tokens.append((token, tag, lemma))

            # 2. THE VERB FILTER (Crucial: Compare LEMMA, not TOKEN)
            # We look for the target lemma (e.g. 'run') matching the current word's lemma (e.g. 'ran'->'run')
            # AND check if it has a Verb tag (VB, VBD, VBG, etc.)
            if lemma == target_lemma.lower() and tag.startswith('V'):
                target_is_verb = True
                
                # 3. THE CONTEXT WINDOW (For AI Judge/Idiom analysis)
                # Grab the next 5 tokens 
                context_start = i + 1
                context_end = min(len(tokens), context_start + 5)
                context_window = tokens[context_start:context_end]
                
                # We stop after the first valid verb instance to keep the context window specific
                break 

        # 4. METRIC: CONTRACTION RATIO 
        num_contractions = sum(1 for t in tokens if "'" in t)
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'lemmatized': lemmatized_tokens,
            'is_valid_verb': target_is_verb,  # <--- The Gatekeeper for the AI Judge
            'context_window': context_window, 
            'num_contractions': num_contractions,
            'token_count': len(tokens)
        }
    
    def process_all_sentences(self) -> List[Dict[str, Any]]:
        if not self.sentences_data:
            print("No sentences loaded.")
            return []
        
        self.pos_results = []
        
        for idx, item in enumerate(self.sentences_data, 1):
            sentence = item.get('sentence', '')
            target_lemma = item.get('lemma', '') # Get the target word (e.g., 'run')
            
            if not sentence or not target_lemma:
                continue
            
            # Pass the target lemma to the analyzer
            analysis = self.analyze_sentence(sentence, target_lemma)
            
            # THESIS LOGIC: Only keep if it is actually used as a verb
            if analysis['is_valid_verb']:
                result = {
                    'index': idx,
                    'register': item.get('register', 'N/A'),
                    'mood': item.get('mood', 'N/A'),
                    'sentence': sentence,
                    'target_lemma': target_lemma,
                    'source': item.get('Source', 'N/A'),
                    'cefr_level': item.get('CEFR_Target', 'N/A'),
                    'analysis': analysis
                }
                self.pos_results.append(result)
        
        print(f"✓ Processed and Filtered. Kept {len(self.pos_results)} valid verb sentences.")
        return self.pos_results
    
    def display_analysis(self, max_items: int = 5) -> None:
        """
        Display POS analysis results in a readable format
        
        Args:
            max_items: Maximum number of sentences to display
        """
        if not self.pos_results:
            print("No results to display. Run process_all_sentences() first.")
            return
        
        for result in self.pos_results[:max_items]:
            print("\n" + "="*80)
            print(f"Index: {result['index']}")
            print(f"Register: {result['register']} | Mood: {result['mood']} | CEFR: {result['cefr_level']}")
            print(f"Sentence: {result['sentence']}")
            print(f"Target Lemma: {result['target_lemma']}")
            print("\nPOS Tags:")
            for token, pos in result['analysis']['pos_tags']:
                print(f"  {token:15} → {pos}")
            print("\nLemmatized Form:")
            for token, pos, lemma in result['analysis']['lemmatized']:
                print(f"  {token:15} ({pos:4}) → {lemma}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed sentences
        
        Returns:
            Dictionary with various statistics
        """
        if not self.pos_results:
            return {}
        
        # Count POS tags
        pos_tag_counts = {}
        total_tokens = 0
        
        for result in self.pos_results:
            for token, pos in result['analysis']['pos_tags']:
                pos_tag_counts[pos] = pos_tag_counts.get(pos, 0) + 1
                total_tokens += 1
        
        # Count registers and moods
        register_counts = {}
        mood_counts = {}
        cefr_counts = {}
        
        for result in self.pos_results:
            register = result['register']
            register_counts[register] = register_counts.get(register, 0) + 1
            
            mood = result['mood']
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
            cefr = result['cefr_level']
            cefr_counts[cefr] = cefr_counts.get(cefr, 0) + 1
        
        return {
            'total_sentences': len(self.pos_results),
            'total_tokens': total_tokens,
            'avg_tokens_per_sentence': total_tokens / len(self.pos_results),
            'unique_pos_tags': len(pos_tag_counts),
            'pos_tag_distribution': dict(sorted(pos_tag_counts.items(), key=lambda x: x[1], reverse=True)),
            'register_distribution': register_counts,
            'mood_distribution': mood_counts,
            'cefr_distribution': cefr_counts
        }
    
    def display_statistics(self) -> None:
        """Display comprehensive statistics about the corpus"""
        stats = self.get_statistics()
        
        if not stats:
            print("No statistics available. Run process_all_sentences() first.")
            return
        
        print("\n" + "="*80)
        print("CORPUS STATISTICS")
        print("="*80)
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Average Tokens per Sentence: {stats['avg_tokens_per_sentence']:.2f}")
        print(f"Unique POS Tags: {stats['unique_pos_tags']}")
        
        print("\n--- POS Tag Distribution (Top 15) ---")
        for pos, count in list(stats['pos_tag_distribution'].items())[:15]:
            percentage = (count / stats['total_tokens']) * 100
            print(f"  {pos:6} : {count:6} ({percentage:5.2f}%)")
        
        print("\n--- Register Distribution ---")
        for register, count in sorted(stats['register_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {register:15} : {count:6}")
        
        print("\n--- Mood Distribution ---")
        for mood, count in sorted(stats['mood_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {mood:15} : {count:6}")
        
        print("\n--- CEFR Level Distribution ---")
        for cefr, count in sorted(stats['cefr_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cefr:6} : {count:6}")
    
    def export_results(self, output_file: str = 'outputs/pos_tagging_results.json') -> bool:
        """
        Export POS tagging results to a JSON file
        
        Args:
            output_file: Path to the output JSON file
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert analysis data to JSON-serializable format
            export_data = []
            for result in self.pos_results:
                export_result = {
                    'index': result['index'],
                    'register': result['register'],
                    'mood': result['mood'],
                    'sentence': result['sentence'],
                    'target_lemma': result['target_lemma'],
                    'source': result['source'],
                    'cefr_level': result['cefr_level'],
                    'analysis': {
                        'tokens': result['analysis']['tokens'],
                        'pos_tags': [{'token': t, 'pos': p} for t, p in result['analysis']['pos_tags']],
                        'lemmatized': [{'token': t, 'pos': p, 'lemma': l} for t, p, l in result['analysis']['lemmatized']],
                        'token_count': result['analysis']['token_count']
                    }
                }
                export_data.append(export_result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results exported to '{output_file}'")
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
    
    def get_lemma_occurrences(self, target_lemma: str) -> List[Dict[str, Any]]:
        """
        Find all occurrences of a specific lemma in the corpus
        
        Args:
            target_lemma: The lemma to search for
            
        Returns:
            List of sentences containing the lemma
        """
        occurrences = []
        
        for result in self.pos_results:
            lemmatized = result['analysis']['lemmatized']
            for token, pos, lemma in lemmatized:
                if lemma.lower() == target_lemma.lower():
                    occurrences.append({
                        'sentence': result['sentence'],
                        'register': result['register'],
                        'mood': result['mood'],
                        'cefr_level': result['cefr_level'],
                        'pos_tag': pos,
                        'token': token
                    })
                    break
        
        return occurrences


def main():
    """Main execution function"""
    print("POS TAGGER - Loading and Processing Sentences")
    print("="*80)
    
    # Initialize tagger
    tagger = POSTagger('outputs/intermediate_sentences.json')
    
    # Load sentences
    if not tagger.load_sentences():
        return
    
    # Process all sentences
    tagger.process_all_sentences()
    
    # Display sample results
    print("\n--- SAMPLE RESULTS (First 3 Sentences) ---")
    tagger.display_analysis(max_items=3)
    
    # Display statistics
    tagger.display_statistics()
    
    # Export results
    tagger.export_results('outputs/pos_tagging_results.json')
    
    # Example: Find a specific lemma
    print("\n" + "="*80)
    print("SEARCHING FOR LEMMA 'take'")
    print("="*80)
    take_occurrences = tagger.get_lemma_occurrences('take')
    print(f"Found {len(take_occurrences)} occurrences of 'take'")
    
    for i, occurrence in enumerate(take_occurrences[:5], 1):
        print(f"\n{i}. {occurrence['sentence']}")
        print(f"   Token: {occurrence['token']} | POS: {occurrence['pos_tag']} | Register: {occurrence['register']}")


if __name__ == '__main__':
    main()
