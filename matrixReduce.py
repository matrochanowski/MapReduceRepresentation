import numpy as np


class MapReduce:
    def __init__(self, computers: list):
        self.computers = computers
        self.numComputers = len(computers)
        self.splitted = None
        self.finished = None
        self.final = None
        self.final_size = (0, 0)

    def split_multiply(self, matrix1, matrix2):
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError("Incorrect dimensions")

        self.final_size = (matrix1.shape[0], matrix2.shape[1])

        self.splitted = dict()

        tasks = [(i, j) for i in range(self.final_size[0]) for j in range(self.final_size[1])]

        for idx, (i, j) in enumerate(tasks):
            computer_idx = idx % self.numComputers
            if computer_idx not in self.splitted:
                self.splitted[computer_idx] = []
            self.splitted[computer_idx].append((i, j, matrix1[i, :], matrix2[:, j]))

    def map(self):
        for i, tasks in self.splitted.items():
            current_machine = self.computers[i]
            for task in tasks:
                current_machine.post_data(task)
                current_machine.compute_multiplication()

    def reduce(self):
        final = np.empty(self.final_size)

        for current_machine in self.computers:
            for machine_result in current_machine.results:
                index = machine_result[0]
                final[index] = machine_result[1]

        self.final = final


class Computer:
    def __init__(self):
        self.data = []
        self.results = []

    def post_data(self, data):
        self.data.append(data)

    def compute_multiplication(self):
        for data in self.data:
            index = (data[0], data[1])
            row = data[2]
            column = data[3]
            result = np.dot(row, column)
            self.results.append((index, result))


def test_classes():
    computers = [Computer() for _ in range(100_000)]
    mr = MapReduce(computers)
    rows1, cols1, cols2 = 300, 50, 500
    macierz1, macierz2 = generate_matrices(rows1, cols1, cols2)
    mr.split_multiply(macierz1, macierz2)
    mr.map()
    mr.reduce()
    print(mr.final.shape)
    print(mr.final)


def generate_matrices(rows1, cols1, cols2):
    matrix1 = np.random.randint(0, 10, (rows1, cols1))
    matrix2 = np.random.randint(0, 10, (cols1, cols2))
    return matrix1, matrix2


# macierz1 = np.array([[1, 2], [3, 4], [5, 6]])
# macierz2 = np.array([[1, 2, 5, 9], [3, 4, 5, 8]])
test_classes()
