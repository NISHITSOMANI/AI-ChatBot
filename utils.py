import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load spaCy model globally

class QnA:
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, filepath='AI-Train-Data.json'):
        self.filepath = filepath
        self.data = []
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.preprocessed_questions = []
        self.load_qna()  # Load once when the class is instantiated

    def load_qna(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as file:
                self.data = json.load(file)
        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        self._initialize_vectorizer()

    def _initialize_vectorizer(self):
        if self.questions:  # Initialize only if questions are available
            self.vectorizer = TfidfVectorizer().fit(self.questions)
            self.preprocessed_questions = [self.nlp(q.lower()) for q in self.questions]

    def save_qna(self, new_qna):
        self.data.append(new_qna)
        with open(self.filepath, 'w') as file:
            json.dump(self.data, file, indent=2)
        # After saving, update questions and answers in-memory without reloading the entire file
        self.questions.append(new_qna['question'])
        self.answers.append(new_qna['answer'])
        # Reinitialize vectorizer only when new data is added
        self._initialize_vectorizer()

    def is_greeting(self, user_input):
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon']
        user_input = user_input.lower()
        return any(greet in user_input for greet in greetings)

    def find_answer(self, user_question, threshold=0.7):
        if not self.questions:
            return None

        user_vector = self.vectorizer.transform([user_question])
        similarity = cosine_similarity(user_vector, self.vectorizer.transform(self.questions))
        best_match_idx = similarity.argmax()
        best_score = similarity[0, best_match_idx]

        if best_score >= threshold:
            return self.answers[best_match_idx]
        else:
            return self.fallback_matching(user_question)

    def fallback_matching(self, user_question):
        user_doc = self.nlp(user_question.lower())
        for idx, q_doc in enumerate(self.preprocessed_questions):
            common_tokens = set(token.text for token in user_doc if not token.is_stop) & set(token.text for token in q_doc if not token.is_stop)
            if len(common_tokens) >= 2:
                return self.answers[idx]
        return None
