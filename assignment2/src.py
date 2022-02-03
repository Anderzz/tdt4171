import numpy as np

def normalize(A):
    return A/np.sum(A)

def forward(T, O, f):
    """
    equation 14.12

    Args:
        T (ndarray):  transition matrix
        O (ndarray)): model matrix
        f (ndarray):  message

    """
    return normalize(O @ T.T @ f)

def backward(T, O, f):

    return normalize(T @ O @ f)

#define transition matrix T and model matrices O[0] & O[1]
T = np.array([[0.7, 0.3],
              [0.3, 0.7]])

O = np.array([[[0.9, 0.0],
               [0.0, 0.2]],

               [[0.1, 0.0],
                [0.0, 0.8]]])



def main():

    #our observations
    evidence = np.array([1, 1, 0, 1, 1])

    #for task2
    f = np.array([0.5, 0.5])

    #for task3
    b = np.array([1.0, 1.0])
    f_list = []
    b_list = []
    f_list.append(f)
    b_list.append(b)
    print(f"\nTask 2:\nDay 0 [rain / not_rain]: {f}")

    ############# Task 2 #############
    for day, e in enumerate(evidence):
        if e:     #use the first model matrix
            f = forward(T, O[0], f)
        if not e: #use the second model matrix
            f = forward(T, O[1], f)
        f_list.append(f)
        print(f"Day {day+1} [rain / not_rain]: {f}")
    ############# \Task 2 #############

    ############# Task 3 #############
    print(f"\n\nTask 3: Backward pass")
    for day, e in enumerate(evidence):
        if e:     #use the first model matrix
            b = backward(T, O[0], b)
        if not e: #use the second model matrix
            b = backward(T, O[1], b)
        b_list.append(b)
        print(f"Day {day+1} [rain / not_rain]: {b}")

    #Compute the smoothed probability values with the reversed b's
    day = 0
    print(f"\n\nTask 3: Smoothed values")
    for f, b in zip(f_list, b_list[::-1]):
        day += 1
        res = normalize(f*b)
        print(f"Day {day} [rain / not_rain]: {res}")
    ############# \Task 3 #############

if __name__ == "__main__":
    main()