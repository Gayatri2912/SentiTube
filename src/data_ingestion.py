"""
Data Ingestion Module for SentiTube
This script creates a synthetic dataset of YouTube comments for demonstration
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_PATH, PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Class to handle data ingestion for YouTube comments"""
    
    def __init__(self):
        self.params = PARAMS['data_ingestion']
        self.sample_size = self.params['sample_size']
        self.random_state = self.params['random_state']
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic YouTube comments data for demonstration
        
        Returns:
            DataFrame with comments and sentiment labels
        """
        logger.info("Generating synthetic YouTube comments data...")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Sample positive comments
        positive_comments = [
            "This is amazing! Best video ever!",
            "Absolutely love this content! Keep it up!",
            "Fantastic work! Really helpful and informative.",
            "This is exactly what I was looking for. Thank you!",
            "Outstanding video! Very well explained.",
            "Brilliant! You're the best!",
            "Wow, this is incredible! Love your channel!",
            "Perfect! This helped me so much!",
            "Excellent content as always!",
            "Amazing tutorial! Very clear and easy to follow.",
            "Great job! Looking forward to more videos like this.",
            "Superb quality content! Subscribed!",
            "This is gold! Thank you for sharing!",
            "Wonderful explanation! Finally understood it.",
            "You're a lifesaver! This helped me pass my exam!",
            "Best channel on YouTube for this topic!",
            "Love your teaching style! So engaging!",
            "This deserves more views! Absolutely brilliant!",
            "Thank you so much! You explained it perfectly!",
            "Awesome content! Keep making such videos!",
        ]
        
        # Sample negative comments
        negative_comments = [
            "This is terrible. Waste of time.",
            "Worst video I've ever seen. Disappointing.",
            "Complete garbage. Don't waste your time.",
            "This sucks. Not helpful at all.",
            "Horrible explanation. Couldn't understand anything.",
            "Dislike! This is so boring.",
            "Awful content. Unsubscribing.",
            "This is useless. Terrible tutorial.",
            "Not good at all. Very confusing.",
            "Bad quality video. Too much rambling.",
            "This is stupid. Didn't help me at all.",
            "Terrible. I want my time back.",
            "Boring! Couldn't finish watching.",
            "This is so bad. Poor explanation.",
            "Worst tutorial ever. Completely wrong information.",
            "Hate this. Such a waste.",
            "Disappointing. Expected much better.",
            "This is rubbish. Not recommended.",
            "Awful audio quality. Can't hear anything.",
            "Bad video. Too complicated and confusing.",
        ]
        
        # Sample neutral comments
        neutral_comments = [
            "Okay, I guess.",
            "This is alright.",
            "Hmm, interesting.",
            "Not bad.",
            "It's okay, could be better.",
            "Watched it.",
            "Thanks for the video.",
            "Noted.",
            "Seen it.",
            "Here for the algorithm.",
            "Commenting for engagement.",
            "First!",
            "Who's watching in 2026?",
            "Anyone else here from TikTok?",
            "What's the song in the background?",
            "Part 2 please?",
            "When is the next video?",
            "Can you make a video on XYZ?",
            "What software did you use?",
            "Link in description?",
        ]
        
        # Calculate distribution for each sentiment (40% positive, 30% negative, 30% neutral)
        n_positive = int(self.sample_size * 0.40)
        n_negative = int(self.sample_size * 0.30)
        n_neutral = self.sample_size - n_positive - n_negative
        
        # Generate comments with variations
        comments = []
        sentiments = []
        
        # Generate positive comments (label: 2)
        for _ in range(n_positive):
            base_comment = np.random.choice(positive_comments)
            # Add some variations
            variations = ["", "!", " 😊", " ❤️", " 👍", " 🔥"]
            comment = base_comment + np.random.choice(variations)
            comments.append(comment)
            sentiments.append(2)
        
        # Generate negative comments (label: 0)
        for _ in range(n_negative):
            base_comment = np.random.choice(negative_comments)
            variations = ["", ".", " 😠", " 👎", " 😤"]
            comment = base_comment + np.random.choice(variations)
            comments.append(comment)
            sentiments.append(0)
        
        # Generate neutral comments (label: 1)
        for _ in range(n_neutral):
            base_comment = np.random.choice(neutral_comments)
            variations = ["", ".", " 🤔"]
            comment = base_comment + np.random.choice(variations)
            comments.append(comment)
            sentiments.append(1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'comment': comments,
            'sentiment': sentiments
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Add metadata
        df['comment_length'] = df['comment'].str.len()
        df['word_count'] = df['comment'].str.split().str.len()
        
        logger.info(f"Generated {len(df)} synthetic comments")
        logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save data to CSV file
        
        Args:
            df: DataFrame to save
        """
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Data saved to {RAW_DATA_PATH}")
    
    def run(self) -> None:
        """Execute the data ingestion pipeline"""
        logger.info("Starting data ingestion...")
        
        # Generate synthetic data
        df = self.generate_synthetic_data()
        
        # Save data
        self.save_data(df)
        
        logger.info("Data ingestion completed successfully!")


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
