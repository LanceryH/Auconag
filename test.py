import multiprocessing.dummy as mp

class MyClass:
    def __init__(self):
        self.data = []

    def update_data(self, value):
        """Function that updates the class attribute."""
        print(f"Processing {value}")
        self.data.append(value)  # This is not thread-safe

    def run_threads(self, values):
        """Run the update_data method using multiple threads."""
        with mp.Pool(4) as pool:  # Using 4 threads
            pool.map(self.update_data, values)

# Example usage
obj = MyClass()
obj.run_threads([1, 2, 3, 4, 5])

print(obj.data)  # Might not be consistent due to threading
