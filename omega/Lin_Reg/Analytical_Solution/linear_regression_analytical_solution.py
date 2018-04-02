import statistics as st
import csv


def lr(input_value):
    x = []
    y = []
    l1 = []
    r = []
    cov = []

    # The train.csv should a 2 column file with first column containing all predictor variable i.e., x and the second
    # column with traget variable i.e., y. Sample file is attached in the respository.

    with open("train.csv", 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            x.append(float(line[0]))
            y.append(float(line[1]))
        mean_x = st.mean(x)
        mean_y = st.mean(y)
        for i in range(len(x)):
            l1.append((x[i]-mean_x)*(y[i]-mean_y))
            r.append((x[i]-mean_x)**2)
        print (l1)
        print (r)
        cov = sum(l1)/sum(r)
        b1 = cov
        b0 = mean_y - b1*mean_x
        predicted_value = b0 + b1*input_value
        return predicted_value


print (lr(77))
