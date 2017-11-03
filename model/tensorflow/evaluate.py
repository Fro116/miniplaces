path = 'logs/binary.log'
classes = 10
test_size = 1000
match = "Evaluation Finished"
first_label_index = 5
second_label_index = 9
data_line = False
predictions = [[0 for x in range(classes)] for test in range(test_size)]
with open(path) as f:
    for line in f:
        if line.startswith(match):
            fields = line.split()
            first_label = int(fields[first_label_index][:-1])
            second_label = int(fields[second_label_index])
            data_line = True
        elif data_line:
            data_line = False
            data = line[1:-2].split(', ')
            for i in range(test_size):
                if data[i] == '0':
                    predictions[i][first_label] += 1
                else:
                    predictions[i][second_label] += 1
    print(predictions)

