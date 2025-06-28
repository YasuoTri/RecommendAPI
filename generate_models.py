#!/usr/bin/env python3
"""
Script to generate required model files for the Course Recommendation System.
Run this before starting the FastAPI server.
"""

import os
import sys
from CourseRecommendationSystem import update_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'Data/udemy_courses.csv',
        'Data/ratings.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.error("Please ensure the following files exist:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("All required data files found.")
    return True

def main():
    """Main function to generate model files."""
    logger.info("Starting model generation process...")
    
    # Check if data files exist
    if not check_data_files():
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        logger.info("Created models/ directory")
    
    try:
        # Generate models
        logger.info("Generating models... This may take a few minutes.")
        success = update_model('Data/udemy_courses.csv')
        
        if success:
            logger.info("Models generated successfully!")
            logger.info("Generated files:")
            model_files = [
                'models/courses.pkl',
                'models/tfidf_vectorizer.pkl',
                'models/tfidf_matrix.pkl',
                'models/svd_model.pkl',
                'models/user_item_matrix.pkl'
            ]
            
            for file_path in model_files:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    logger.info(f"  - {file_path} ({size:.2f} MB)")
                else:
                    logger.warning(f"  - {file_path} (NOT FOUND)")
            
            logger.info("You can now start the FastAPI server with: uvicorn main:app --reload")
        else:
            logger.error("Model generation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during model generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
