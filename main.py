# main.py

from data_loader import load_and_preprocess
from model_trainer import train_and_evaluate

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess()

    print("Training and evaluating model...")
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
