import sys
import subprocess


def run_train_model():
    # Run the train_model.py script
    subprocess.run(['python', 'train_model.py'])


def run_predict(user_id, book_ids, top_n):

    subprocess.run(['python', 'predict.py', str(user_id)] + book_ids + [str(top_n)])


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train/predict]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        run_train_model()

    elif mode == "predict":

        user_id = input("Enter user ID: ")

        books_input = input("Enter book IDs separated by commas: ")

        book_ids = books_input.split(',')

        top_n = int(input("Enter number of recommendations: "))

        run_predict(user_id, book_ids, top_n)

    else:
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
