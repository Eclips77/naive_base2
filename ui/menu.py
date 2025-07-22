from naive_bayes.app import App


class Menu:
    """Simple CLI menu for interacting with :class:`App`."""

    def __init__(self) -> None:
        self.app = App()

    def display(self) -> None:
        """Show the menu loop until the user exits."""
        while True:
            print("\n--- Naive Bayes Classifier Menu ---")
            print("1. Load & clean data")
            print("2. Train model")
            print("3. Evaluate model")
            print("4. Classify new record")
            print("5. Exit")

            choice = input("Choose an option (1-5): ")

            if choice == "1":
                file_name = input("Enter CSV file name: ")
                self.app.load_and_clean(file_name)
                target_column = input("Enter target column name: ")
                self.app.set_target_column(target_column)
            elif choice == "2":
                print(self.app.train_model())
            elif choice == "3":
                print(self.app.evaluate_model())
            elif choice == "4":
                record_input = input(
                    "Enter record as key=value pairs separated by commas: "
                )
                record = dict(pair.split("=") for pair in record_input.split(","))
                print(self.app.classify_record(record))
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Try again.")


# if __name__ == "__main__":
    # menu = Menu()
    # menu.display()

