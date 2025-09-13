"""
internship_recommender.py

Content-based Internship Recommendation System
Using: TF-IDF (text) + OneHotEncoder (categorical) + MinMaxScaler (numerical) + Cosine Similarity

Requirements:
- pandas
- numpy
- scikit-learn
- scipy
- joblib
- (optional) flask for quick integration example
"""

import os
import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternshipRecommender:
    def __init__(self,
                 text_cols: List[str] = None,
                 categorical_cols: List[str] = None,
                 numeric_cols: List[str] = None,
                 tfidf_kwargs: Dict[str, Any] = None):
        """
        Parameters:
        - text_cols: list of column names to treat as text (e.g., ['Job Title', 'Company', 'Cities', 'Skills'])
        - categorical_cols: list of column names to one-hot encode (e.g., ['Cities'])
        - numeric_cols: list of numeric columns that will be scaled (e.g., ['Stipend','Duration'])
        - tfidf_kwargs: dict of kwargs passed to TfidfVectorizer
        """
        # default candidates (will filter against df columns)
        self.default_text_cols = ['Company', 'Company Name', 'Job Title', 'JobTitle', 'Title', 'Cities', 'City', 'Skills', 'Description', 'Requirements']
        self.default_categorical_cols = ['Cities', 'City', 'Location', 'State', 'States']
        self.default_numeric_cols = ['Stipend', 'Salary', 'Duration', 'Months']

        self.text_cols = text_cols
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

        self.tfidf_kwargs = tfidf_kwargs or {'min_df': 1, 'ngram_range': (1, 2), 'stop_words': 'english', 'max_features': 20000}

        # Models / transformers (will be fitted in fit())
        self.tfidf = TfidfVectorizer(**self.tfidf_kwargs)
        self.cat_encoder = None  # OneHotEncoder(handle_unknown='ignore', sparse=True)
        self.scaler = None  # MinMaxScaler()
        self.feature_matrix = None  # sparse matrix of dataset features
        self.df = None  # original dataframe
        self.fitted = False

        # Keep track of actual columns used
        self._used_text_cols = []
        self._used_cat_cols = []
        self._used_num_cols = []

    def _choose_columns(self, df: pd.DataFrame):
        # Determine which columns exist in df and choose defaults if not provided
        if self.text_cols is None:
            self._used_text_cols = [c for c in self.default_text_cols if c in df.columns and df[c].dtype == object]
        else:
            self._used_text_cols = [c for c in self.text_cols if c in df.columns]

        if self.categorical_cols is None:
            self._used_cat_cols = [c for c in self.default_categorical_cols if c in df.columns]
        else:
            self._used_cat_cols = [c for c in self.categorical_cols if c in df.columns]

        if self.numeric_cols is None:
            self._used_num_cols = [c for c in self.default_numeric_cols if c in df.columns]
        else:
            self._used_num_cols = [c for c in self.numeric_cols if c in df.columns]

        logger.info("Using text columns: %s", self._used_text_cols)
        logger.info("Using categorical columns: %s", self._used_cat_cols)
        logger.info("Using numeric columns: %s", self._used_num_cols)

    @staticmethod
    def _combine_text_row(row: pd.Series, text_cols: List[str]) -> str:
        # Concatenate textual columns into a single string; drop NaN and cast to str
        pieces = []
        for col in text_cols:
            val = row.get(col, "")
            if pd.isna(val):
                continue
            pieces.append(str(val))
        return " ".join(pieces).strip()

    def _create_text_series(self, df: pd.DataFrame) -> pd.Series:
        # Returns pandas Series with combined text per row
        return df.apply(lambda r: self._combine_text_row(r, self._used_text_cols), axis=1)

    def fit(self, df: pd.DataFrame):
        """
        Fit transformers and build dataset feature matrix.

        - df: pandas DataFrame containing internships
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        # Copy and basic cleaning
        self.df = df.copy().reset_index(drop=True)
        self.df.fillna("", inplace=True)  # simple strategy: empty string for missing textual/categorical cells

        # choose columns dynamically if user didn't provide
        self._choose_columns(self.df)

        # 1) TF-IDF on combined text
        text_series = self._create_text_series(self.df)
        logger.info("Fitting TF-IDF on text data (n_rows=%d)...", len(text_series))
        tfidf_matrix = self.tfidf.fit_transform(text_series)  # sparse matrix shape (n_samples, n_tfidf_features)
        logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

        # 2) One-hot encode categorical columns
        if len(self._used_cat_cols) > 0:
            # create a dataframe with only categorical cols (cast to str)
            cat_df = self.df[self._used_cat_cols].astype(str).map(lambda x: x.strip() if isinstance(x, str) else str(x))
            self.cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            cat_sparse = self.cat_encoder.fit_transform(cat_df)
            logger.info("Categorical one-hot shape: %s", cat_sparse.shape)
        else:
            cat_sparse = None

        # 3) Scale numeric columns
        if len(self._used_num_cols) > 0:
            num_df = self.df[self._used_num_cols].astype(float)  # may raise if conversion fails
            self.scaler = MinMaxScaler()
            num_scaled = self.scaler.fit_transform(num_df)  # dense ndarray (n_samples, n_num_cols)
            # convert numeric dense to sparse for stacking
            num_sparse = csr_matrix(num_scaled)
            logger.info("Numeric scaled shape: %s", num_scaled.shape)
        else:
            num_sparse = None

        # 4) Stack horizontally: [tfidf | cat_onehot | num_scaled]
        mats = [tfidf_matrix]
        if cat_sparse is not None:
            mats.append(cat_sparse)
        if num_sparse is not None:
            mats.append(num_sparse)

        self.feature_matrix = hstack(mats).tocsr()
        logger.info("Combined feature matrix shape: %s", self.feature_matrix.shape)
        self.fitted = True
        return self

    def save_model(self, model_path: str = "internship_recommender_model.pkl"):
        """
        Save the trained model to disk using joblib.
        
        Parameters:
        - model_path: Path where to save the model file
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit(df) before saving.")
        
        try:
            joblib.dump(self, model_path)
            logger.info("Model saved successfully to %s", model_path)
        except Exception as e:
            logger.exception("Failed to save model: %s", e)
            raise

    @classmethod
    def load_model(cls, model_path: str):
        """
        Load a trained model from disk using joblib.
        
        Parameters:
        - model_path: Path to the saved model file
        
        Returns:
        - InternshipRecommender: Loaded model instance
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            loaded_model = joblib.load(model_path)
            logger.info("Model loaded successfully from %s", model_path)
            return loaded_model
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            raise

    def _prepare_user_vector(self, user_profile: Dict[str, Any]) -> csr_matrix:
        """
        Convert user_profile into the same feature-space vector (1 x D sparse matrix).
        user_profile is a dict e.g.:
            {
              'Skills': 'Python, ML',
              'Preferred Cities': 'Mumbai, Pune',
              'Min Stipend': 12000,
              'Duration': 6,
              'Job Title': 'Data Science Intern'
            }
        Mapping user_profile keys to dataframe columns is flexible:
           - for text fields we look into self._used_text_cols
           - for categorical we look into self._used_cat_cols
           - for numeric we look into self._used_num_cols
        If user_profile contains free-form keys (like 'Skills' or 'Preferred Cities'),
        we will try to place them where they make most sense.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit(df) before recommending.")

        # Build a one-row DataFrame to reuse transformation logic
        sample = {col: "" for col in (self._used_text_cols + self._used_cat_cols + self._used_num_cols)}
        # try to intelligently map common keys
        # direct mapping if exact keys exist
        for k, v in user_profile.items():
            if k in sample:
                sample[k] = v
            # some common alternative keys:
            elif k.lower() in ['skills', 'skill', 'technologies']:
                # append into text fields; prefer a 'Skills' text column if present, else append to first text col
                if 'Skills' in sample:
                    sample['Skills'] = v
                else:
                    # append to the first text column
                    if self._used_text_cols:
                        sample[self._used_text_cols[0]] += " " + str(v)
            elif k.lower() in ['preferred cities', 'preferred city', 'cities', 'city', 'locations', 'location']:
                # put into first categorical col if available else into text
                if self._used_cat_cols:
                    sample[self._used_cat_cols[0]] = v
                elif self._used_text_cols:
                    sample[self._used_text_cols[0]] += " " + str(v)
            elif k.lower() in ['min stipend', 'stipend', 'salary', 'min_salary', 'expected stipend']:
                # numeric mapping
                if 'Stipend' in sample:
                    try:
                        sample['Stipend'] = float(v)
                    except Exception:
                        sample['Stipend'] = 0.0
                else:
                    # if no stipend column in dataset, ignore numeric
                    pass
            elif k.lower() in ['duration', 'months']:
                if 'Duration' in sample:
                    try:
                        sample['Duration'] = float(v)
                    except Exception:
                        sample['Duration'] = 0.0
            else:
                # fallback: put any free-text into first text column (if exists)
                if self._used_text_cols:
                    sample[self._used_text_cols[0]] += " " + str(v)

        user_df = pd.DataFrame([sample])
        user_df.fillna("", inplace=True)

        # 1) text -> tfidf
        user_text = self._create_text_series(user_df)  # length 1
        user_tfidf = self.tfidf.transform(user_text)

        # 2) categorical -> onehot transform (handle_unknown='ignore' set earlier)
        if self._used_cat_cols:
            cat_df = user_df[self._used_cat_cols].astype(str).map(lambda x: x.strip() if isinstance(x, str) else str(x))
            user_cat = self.cat_encoder.transform(cat_df)
        else:
            user_cat = None

        # 3) numeric -> scale (if provided in user_profile)
        if self._used_num_cols:
            try:
                num_df = user_df[self._used_num_cols].astype(float)
            except Exception:
                # if conversion fails, replace with zeros
                num_df = pd.DataFrame(np.zeros((1, len(self._used_num_cols))), columns=self._used_num_cols)
            user_num_scaled = self.scaler.transform(num_df)  # shape (1, n_num)
            user_num_sparse = csr_matrix(user_num_scaled)
        else:
            user_num_sparse = None

        mats = [user_tfidf]
        if user_cat is not None:
            mats.append(user_cat)
        if user_num_sparse is not None:
            mats.append(user_num_sparse)

        user_vec = hstack(mats).tocsr()
        # Ensure same number of columns (if dataset has columns but user doesn't produce some features)
        # hstack with fitted encoders/scalers will produce the right shape
        return user_vec

    def recommend(self, user_profile: Dict[str, Any], top_n: int = 5, return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend top_n internships for given user_profile.

        Returns a list of dicts with row index, similarity score, and original row data (columns).
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit(df) before recommending.")

        try:
            user_vec = self._prepare_user_vector(user_profile)
        except Exception as e:
            logger.exception("Failed to create user vector: %s", e)
            raise

        # compute cosine similarity between user vector and all dataset rows
        # cosine_similarity returns (1, n_samples)
        sims = cosine_similarity(user_vec, self.feature_matrix).flatten()
        if np.all(np.isnan(sims)):
            raise RuntimeError("All similarities are NaN - check input and fitted transforms.")

        # Get top indices
        top_idx = np.argsort(-sims)[:top_n]  # descending order
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            row = self.df.iloc[idx].to_dict()
            row_result = {
                "index": int(idx),
                "similarity": score,
                "row": row
            }
            results.append(row_result)
        return results

    # Flask-ready integration function:
    def get_recommendations_api(self, user_profile: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Simple wrapper that returns JSON-serializable results for web APIs.
        """
        recs = self.recommend(user_profile, top_n=top_n)
        # return in API-friendly format
        payload = {
            "requested_profile": user_profile,
            "top_n": top_n,
            "recommendations": recs
        }
        return payload


# --------------------------
# Training and saving functions
# --------------------------

def _parse_stipend(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    if s.lower() == 'unpaid':
        return 0.0
    digits = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
    try:
        return float(digits) if digits else 0.0
    except Exception:
        return 0.0

def _parse_duration_to_months(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    # Extract number
    num_str = ''.join(ch for ch in s if (ch.isdigit() or ch == '.'))
    try:
        num = float(num_str) if num_str else 0.0
    except Exception:
        num = 0.0
    s_lower = s.lower()
    if 'week' in s_lower:
        return num / 4.0  # approx convert weeks to months
    return num  # treat as months otherwise

def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize company name column
    if 'Company' not in df.columns and 'Company Name' in df.columns:
        df['Company'] = df['Company Name']
    # Numeric conversions
    if 'Stipend' in df.columns:
        df['Stipend'] = df['Stipend'].apply(_parse_stipend)
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(_parse_duration_to_months)
    return df

def load_sample_df():
    """
    Provide fallback sample DataFrame if CSV path not provided.
    This mirrors the small dataset provided in the prompt.
    """
    sample = pd.DataFrame({
        "Stipend": [10000.0, 12000.0, 15000.0, 12333.333333, 12333.333333],
        "Duration": [6.0, 6.0, 7.0, 1.0, 10.0],
        "Cities": ["Mumbai", "Delhi", "Pune", "Noida", "Bangalore"],
        # Add a 'Job Title' and 'Company' so TF-IDF has some text content
        "Job Title": ["Data Science Intern", "ML Research Intern", "Backend Intern", "Frontend Intern", "Data Analyst Intern"],
        "Company": ["A Corp", "B Labs", "C Systems", "D Designs", "E Analytics"],
        "Skills": ["Python, ML, pandas", "PyTorch, ML, research", "Python, Django, REST", "React, CSS, JS", "SQL, Excel, Python"]
    })
    return sample

def load_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            logger.info("Loaded data from %s (shape=%s)", csv_path, df.shape)
            df = _clean_dataset(df)
            logger.info("Cleaned dataset shape: %s", df.shape)
            return df
        except Exception as e:
            logger.exception("Failed to read CSV at %s: %s", csv_path, e)
            raise
    else:
        logger.warning("CSV path not provided or not found; using sample dataset.")
        return load_sample_df()

def train_and_save_model(csv_path: str = None, model_path: str = "internship_recommender_model.pkl"):
    """
    Train and save the internship recommender model.
    This should be run separately from your Flask app.
    
    Parameters:
    - csv_path: Path to the internship CSV file
    - model_path: Path where to save the trained model
    """
    print("Loading and preparing data...")
    df = load_data(csv_path)
    
    print("Training model...")
    recommender = InternshipRecommender()
    recommender.fit(df)
    
    print("Saving model...")
    recommender.save_model(model_path)
    
    print(f"Model training complete! Saved to: {model_path}")
    return recommender

if __name__ == "__main__":
    # Training script - run this once to train and save the model
    default_csv = os.path.join(os.path.dirname(__file__), 'Internship.csv')
    model_path = "internship_recommender_model.pkl"
    
    # Train and save model
    recommender = train_and_save_model(
        csv_path=default_csv if os.path.exists(default_csv) else None,
        model_path=model_path
    )
    
    # Test the saved model by loading it
    print("\nTesting saved model...")
    loaded_recommender = InternshipRecommender.load_model(model_path)
    
    # Example test cases
    test_profiles = [
        {
            "Skills": "Python, Machine Learning, pandas",
            "Preferred Cities": "Thiruvananthapuram, Bangalore",
            "Min Stipend": 10000,
            "Duration": 6,
            "Job Title": "Machine Learning Intern"
        },
        {
            "Skills": "React, CSS, JavaScript",
            "Preferred Cities": "Chandigarh, Calicut",
            "Min Stipend": 10000,
            "Duration": 6,
            "Job Title": "Web Development Intern"
        }
    ]

    for i, user_profile in enumerate(test_profiles, start=1):
        recs = loaded_recommender.recommend(user_profile, top_n=3)
        print(f"\nTest case {i} - Top recommendations:")
        for r in recs:
            idx = r['index']
            sim = r['similarity']
            row = r['row']
            print(f"Index: {idx} | Similarity: {sim:.4f} | Company: {row.get('Company','-')} | Title: {row.get('Job Title','-')} | City: {row.get('Cities','-')} | Stipend: {row.get('Stipend','-')} | Duration(mo): {row.get('Duration','-')}")
